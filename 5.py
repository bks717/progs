#pip -q install gensim

import gensim.downloader as api

# Load model
wv = api.load("glove-wiki-gigaword-50")

# Input
seed = input("Enter a seed word: ")

try:
    sw = [w for w, _ in wv.most_similar(seed, topn=5)]
    print("Similar words:", sw)

    # Paragraph using given templates
    para = " ".join([
        f"The {seed} was surrounded by {sw[0]} and {sw[1]}.",
        f"People often associate {seed} with {sw[2]} and {sw[3]}.",
        f"In the land of {seed}, {sw[4]} was a common sight.",
        f"A story about {seed} would be incomplete without {sw[1]} and {sw[3]}."
    ])

    print("\nGenerated Paragraph:\n", para)

except KeyError:
    print("Word not found")