import pandas as pd
import json
from sentence_transformers import SentenceTransformer
import numpy as np

# Load the JSON data
with open('data/classes.json', 'r') as f:
    data = json.load(f)

# Initialize the Sentence Transformer model
model = SentenceTransformer('all-mpnet-base-v2')
max_seq_length = model.get_max_seq_length()
print(f"max_seq_length: {max_seq_length}")

def encode_text(text, model, max_seq_length):
    """Encode long text by chunking and averaging the embeddings."""
    # Split the text into words and chunk them based on max_seq_length
    words = text.split()
    chunks = [" ".join(words[i:i + max_seq_length]) for i in range(0, len(words), max_seq_length)]
    embeddings = [model.encode(chunk) for chunk in chunks]
    return np.mean(embeddings, axis=0)

# Create an empty DataFrame to store the results
df = pd.DataFrame(columns=['label', 'type', 'count', 'embedding'])
rows_list = []

# Iterate through each superclass
for superclass, superclass_data in data.items():
    superclass_embedding = encode_text(superclass, model, max_seq_length)
    rows_list.append({
        'label': superclass,
        'type': 'superclass',
        'count': superclass_data['count'],
        'embedding': superclass_embedding
    })

    # Iterate through each class within the superclass
    for class_name, class_data in superclass_data['classes'].items():
        class_embedding = encode_text(class_name, model, max_seq_length)
        rows_list.append({
            'label': class_name,
            'type': 'class',
            'count': class_data['count'],
            'embedding': class_embedding
        })

# Convert rows_list to DataFrame and concatenate with the main df
df = pd.concat([df, pd.DataFrame(rows_list)], ignore_index=True)

# Save the DataFrame as a pickle file
df.to_pickle('data/classes_embeddings.pkl')
