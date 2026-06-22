# pip install gensim scikit-learn matplotlib
import gensim.downloader as api
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

wv = api.load("word2vec-google-news-300")
# ── Visualize using PCA ───────────────────────────────────────────────────────
def visualize(words):
    vecs   = [wv[w] for w in words]
    points = PCA(n_components=2).fit_transform(vecs)

    plt.scatter(points[:, 0], points[:, 1])
    for i, w in enumerate(words):
        plt.text(points[i, 0], points[i, 1], w)
    plt.title("PCA - Word Embeddings")
    plt.show()
# ── Q1 Functions ──────────────────────────────────────────────────────────────
def word_arithmetic(w1, w2, w3):
    print(f"\n{w1} - {w2} + {w3} =")
    for w, s in wv.most_similar(positive=[w1, w3], negative=[w2], topn=5):
        print(f"  {w}: {s:.4f}")

def word_similarity(w1, w2):
    print(f"\nSimilarity({w1}, {w2}) = {wv.similarity(w1, w2):.4f}")

def most_similar(word):
    print(f"\nMost similar to '{word}':")
    for w, s in wv.most_similar(word, topn=5):
        print(f"  {w}: {s:.4f}")

# ── Run ───────────────────────────────────────────────────────────────────────
word_arithmetic("king", "man", "woman")
word_similarity("cat", "dog")
most_similar("happy")

words = ["computer","laptop","keyboard","mouse","software",
         "hardware","internet","network","data","programming"]

visualize(words)