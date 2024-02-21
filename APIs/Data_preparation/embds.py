import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np
import concurrent.futures
from tqdm import tqdm
import pickle

tqdm.pandas()

def encode_text(args):
    """Encode long text by chunking and averaging the embeddings."""
    text, model, max_seq_length = args
    words = text.split()
    chunks = [" ".join(words[i:i + max_seq_length]) for i in range(0, len(words), max_seq_length)]
    embeddings = [model.encode(chunk) for chunk in chunks]
    return np.mean(embeddings, axis=0)

def calculate_embeddings(texts, model, max_seq_length):
    with concurrent.futures.ProcessPoolExecutor() as executor:
        args = [(text, model, max_seq_length) for text in texts]
        embeddings = list(tqdm(executor.map(encode_text, args), total=len(texts)))
    return embeddings

def main():
    df = pd.read_csv('data/recipes.csv')

    model = SentenceTransformer('all-mpnet-base-v2')
    max_seq_length = model.get_max_seq_length()
    print(f"max_seq_length: {max_seq_length}")

    print("Calculating embeddings for 'title'...")
    df['title_vector'] = calculate_embeddings(df['title'].astype(str).tolist(), model, max_seq_length)

    print("Calculating embeddings for 'title + instructions'...")
    df['title_instructions_vector'] = calculate_embeddings((df['title'].astype(str) + " " + df['instructions'].astype(str)).tolist(), model, max_seq_length)

    print("Calculating embeddings for 'title + instructions + ingredients'...")
    df['all_text_vector'] = calculate_embeddings((df['title'].astype(str) + " " + df['instructions'].astype(str) + " " + df['cleaned_ingredients'].astype(str)).tolist(), model, max_seq_length)

    df.to_pickle("data/embeddings.pkl")

    print("Embeddings calculated and saved to pickle file.")

if __name__ == '__main__':
    main()
