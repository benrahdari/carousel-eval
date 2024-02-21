from flask import Flask, request, jsonify
from flask_restful import Api, Resource
from flask_cors import CORS  
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# Initialize the Flask app and configure its settings
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  
CORS(app)  # Enable CORS for the app
api = Api(app)  # Initialize Flask-RESTful API

# Load the sentence transformer model and data
model = SentenceTransformer('all-mpnet-base-v2')
embeddings_df = pd.read_pickle('data/recipes.pkl')
index = faiss.read_index('recipes.index')

def vectorize(text):
    """Convert text to a vector using the loaded model."""
    return model.encode(text, convert_to_numpy=True)

class RecipeSearch(Resource):
    """Resource for searching recipes based on a text query."""

    def get(self):
        query = request.args.get('query', '')
        if not query:
            return {"message": "Query should not be empty"}, 400

        # Vectorize the query and perform search
        query_vector = vectorize(query)
        distances, indices = index.search(np.array([query_vector]), 50)
        results = embeddings_df.iloc[indices[0]][['title', 'image_name', 'instructions']]
        results['instructions'] = results['instructions'].str[:200]  # Truncate instructions

        return {"results": results.to_dict(orient='records')}

class RecipeSearchCarousels(Resource):
    """Resource for searching recipes and organizing results into carousels based on classes."""

    def get(self):
        query = request.args.get('query', '')
        input_type = request.args.get('input', 'sm')
        if not query:
            return {"message": "Query should not be empty"}, 400

        # Map input type to corresponding vector column
        embedding_column_map = {'sm': 'vector-sm', 'md': 'vector-md', 'lg': 'vector-lg'}

        try:
            query_vector = vectorize(query)
            distances, indices = index.search(np.array([query_vector]), 100)
            results = embeddings_df.iloc[indices[0]]
            results['alltext'] = results['title'].astype(str) + " " + results['instructions'].astype(str)
            embeddings = np.array([row for row in results[embedding_column_map[input_type]]])

            # Process and cluster the results
            class_scores = {}
            for _, recipe in results.iterrows():
                process_class_scores(recipe, class_scores)

            sorted_classes = sort_classes_by_score(class_scores)
            clustered_results = cluster_results_by_class(results, sorted_classes)

            return {"results": clustered_results}

        except Exception as e:
            return {"message": str(e)}, 500

def process_class_scores(recipe, class_scores):
    """Update class scores based on recipe data."""
    class_titles = recipe['class-sm'].split(', ')
    class_scores_list = [float(score) for score in recipe['score-sm'].split(', ')]
    for class_title, class_score in zip(class_titles, class_scores_list):
        class_scores[class_title] = class_scores.get(class_title, 0) + class_score

def sort_classes_by_score(class_scores):
    """Sort classes based on their scores in descending order."""
    return sorted(class_scores.items(), key=lambda x: x[1], reverse=True)

def cluster_results_by_class(results, sorted_classes):
    """Organize results into clusters based on classes."""
    clustered_results = {}
    processed_indices = set()
    for class_title, _ in sorted_classes:
        cluster_data = results[~results.index.isin(processed_indices) & results['class-sm'].str.contains(class_title)]
        cluster_data['instructions'] = cluster_data['instructions'].str[:200]  # Truncate instructions
        if not cluster_data.empty:
            add_cluster(clustered_results, class_title, cluster_data, class_scores, processed_indices)
    return {k: v for k, v in clustered_results.items() if len(v['data']) >= 5}  # Filter clusters with at least 5 items

def add_cluster(clustered_results, class_title, cluster_data, class_scores, processed_indices):
    """Add a new cluster to the results."""
    cluster_info = {
        'name': class_title,
        'score': class_scores[class_title],
        'data': cluster_data.to_dict(orient='records')
    }
    clustered_results[f"cluster_{len(clustered_results)}"] = cluster_info
    processed_indices.update(cluster_data.index)

# Register resources with the API
api.add_resource(RecipeSearch, "/search")
api.add_resource(RecipeSearchCarousels, "/search_carousels")

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0")
