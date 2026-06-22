import gensim.downloader as api
from transformers import pipeline

words = api.load("glove-wiki-gigaword-50")
ai = pipeline("text-generation", model="gpt2")

prompt1 = "Who is the king?"
prompt2 = prompt1.replace("king", words.most_similar("king")[0][0])

ans1 = ai(prompt1, max_new_tokens=40, pad_token_id=50256)[0]['generated_text']
ans2 = ai(prompt2, max_new_tokens=40, pad_token_id=50256)[0]['generated_text']

print("\n--- Results ---")
print("Answer 1:\n", ans1, "\n")
print("Answer 2:\n", ans2)