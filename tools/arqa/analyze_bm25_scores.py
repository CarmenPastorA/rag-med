import os
import sys
import matplotlib.pyplot as plt

# Config paths
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(PROJECT_ROOT)

from tools.arqa.bm25_search import BM25Search

BM25_DIR = os.path.join(PROJECT_ROOT, "data/posteriori_resources/bm25_stuffs")
FASTTEXT_PATH = os.path.join(PROJECT_ROOT, "models/lang_model")
STOPWORDS_PATH = os.path.join(PROJECT_ROOT, "data/priori_resources/stopwords.txt")
PRESERVE_WORDS_PATH = os.path.join(PROJECT_ROOT, "data/priori_resources/preserve_words.txt")

# Init retriever
bm25 = BM25Search(
    directory=BM25_DIR,
    fasttext_model_path=FASTTEXT_PATH,
    stopwords_path=STOPWORDS_PATH,
    preserve_words_path=PRESERVE_WORDS_PATH,
    verbose=False
)

# Example question
question = "¿Qué cama compro en ikea?"

# Run search with scores
results = bm25.search(question, k=100, include_scores=True, translate=True)

# Extract scores
scores = [score for _, score in results]

# Print basic stats
print("Number of results:", len(scores))
print("Max score:", max(scores))
print("Min score:", min(scores))
print("Mean score:", sum(scores) / len(scores))

# Plot histogram
plt.hist(scores, bins=20, color="skyblue", edgecolor="black")
plt.title("Distribución de scores BM25")
plt.xlabel("Score")
plt.ylabel("Número de documentos")
plt.grid(axis="y")
plt.tight_layout()
plt.show()
