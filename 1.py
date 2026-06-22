#pip install gensim
import gensim.downloader as api
# Load pre-trained word vectors
wv = api.load("word2vec-google-news-300")

# 1. Vector Arithmetic (e.g., king - man + woman = queen)
def word_arithmetic(w1, w2, w3):
    result = wv.most_similar(positive=[w1, w3], negative=[w2], topn=5)
    print(f"\n{w1} - {w2} + {w3} =")
    for word, s in result:
        print(f"  {word}: {s:.4f}")

print("Word Arithmetic Examples:")
word_arithmetic("king", "man", "woman")
word_arithmetic("paris", "france", "germany")

# 2. Word Similarity
def word_similarity(w1, w2):
    score = wv.similarity(w1, w2)
    print(f"\nSimilarity({w1}, {w2}) = {score:.4f}")

print("Word Similarity Examples:")
word_similarity("cat", "dog")
word_similarity("music", "art")

# 3. Most Similar Words
def most_similar(word):
    print(f"\nMost similar to '{word}':")
    for w, score in wv.most_similar(word, topn=5):
        print(f"  {w}: {score:.4f}")
        
print("Most Similar Words Examples:")
most_similar("happy")
most_similar("technology")