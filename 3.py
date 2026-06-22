#pip install gensim matplotlib scikit-learn

from gensim.models import Word2Vec
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# ── Medical Domain Corpus ─────────────────────────────────────────────────────
corpus = [
    "patient diagnosed diabetes hypertension",
    "mri scan brain tissue",
    "treatment antibiotics monitoring",
    "symptoms fever fatigue pain",
    "vaccine viral infections",
    "doctor therapy recovery",
    "clinical trial results",
    "surgery procedure",
    "prescription drugs pain",
    "genetic disorder diagnosis"
]

# Preprocess
data = [s.split() for s in corpus]

# ── Train Word2Vec ────────────────────────────────────────────────────────────
model = Word2Vec(data, vector_size=50, window=3, min_count=1, epochs=20)
print("Training complete!")  #not req , but just to confirm that the model has been trained

# ── Visualize using PCA ───────────────────────────────────────────────────────
words  = list(model.wv.index_to_key)
vecs   = [model.wv[w] for w in words]
points = PCA(n_components=2).fit_transform(vecs)

plt.scatter(points[:, 0], points[:, 1])
for i, w in enumerate(words):
    plt.text(points[i, 0], points[i, 1], w, fontsize=9)
plt.show()

# ── Semantic Similarity ───────────────────────────────────────────────────────
def similar_words(word):
    print(f"\nSimilar to '{word}':")
    for w, s in model.wv.most_similar(word, topn=5):
        print(f"  {w}: {s:.4f}")

similar_words("treatment")
similar_words("vaccine")