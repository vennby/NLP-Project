from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import glob

# Step 1: Load the documents
documents = []
filenames = []
for filename in glob.glob("*.txt"):
    with open(filename, 'r', encoding='utf-8') as file:
        content = file.read().split()  # Split text into words
        documents.append(content)
        filenames.append(filename)

# Step 2: Train a Word2Vec model on the documents
# Alternatively, you could load a pre-trained model
model = Word2Vec(documents, vector_size=100, window=5, min_count=1, workers=4)

# Step 3: Represent each document as the average of its word vectors
def document_vector(doc):
    # Filter out words that are not in the model vocabulary
    valid_words = [word for word in doc if word in model.wv]
    if not valid_words:  # Handle empty case if no valid words
        return np.zeros(model.vector_size)
    # Average the word vectors
    return np.mean(model.wv[valid_words], axis=0)

doc_vectors = np.array([document_vector(doc) for doc in documents])

# Step 4: Calculate pairwise cosine similarity between document vectors
similarity_matrix = cosine_similarity(doc_vectors)

# Step 5: Check for similarities above the threshold
threshold = 0.5
print("Warnings for Document Similarities Exceeding Threshold (0.5):")
for i in range(len(documents)):
    for j in range(i + 1, len(documents)):  # Only check upper triangle
        if similarity_matrix[i][j] > threshold:
            print(f"Warning: '{filenames[i]}' and '{filenames[j]}' have similarity {similarity_matrix[i][j]:.2f}")
