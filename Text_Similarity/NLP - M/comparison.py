from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import glob

# Step 1: Load the documents
documents = []
filenames = []
for filename in glob.glob("2024*.txt"):  # Load all .txt files in the directory
    with open(filename, 'r', encoding='utf-8') as file:
        documents.append(file.read())
        filenames.append(filename)

# Step 2: Compute the TF-IDF matrix
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(documents)

# Step 3: Calculate pairwise cosine similarity
similarity_matrix = cosine_similarity(tfidf_matrix)

# Print the similarity matrix
print("Cosine Similarity Matrix:")
print(similarity_matrix)

count = 0
print("Warnings : ")
for i in range(len(similarity_matrix)):
    for j in range(i+1, len(similarity_matrix)):
        if similarity_matrix[i][j] > 0.5:
            count = count +1
            print(filenames[i], "and", filenames[j], "might have been copied!!")

if count==0:
    print("No warnings generated")
