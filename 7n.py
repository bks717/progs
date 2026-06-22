# Install: pip install transformers torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

MODEL_NAME = "sshleifer/distilbart-cnn-12-6"

# ── Step 1: Load model and tokenizer ────────────────────────────────────────
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model     = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

passage = """
Machine learning (ML) is a branch of artificial intelligence (AI) focused on building
systems that learn from data and improve their performance over time without being
explicitly programmed. Traditional programming requires developers to write specific
rules for every task. In contrast, machine learning algorithms identify patterns in
data and use those patterns to make predictions or decisions. 

There are three main types of machine learning: supervised learning, where the model is trained on labeled
data; unsupervised learning, where the model finds hidden patterns in unlabeled data;
and reinforcement learning, where an agent learns by interacting with an environment
and receiving rewards or penalties. 

Machine learning is used in many real-worldapplications such as email spam filtering, recommendation systems, image recognition,
natural language processing, and medical diagnosis. Deep learning, a subset of machine
learning, uses neural networks with many layers to solve complex problems like speech
recognition and autonomous driving. As the volume of data grows and computing power
increases, machine learning is becoming one of the most transformative technologies
of our time.
"""

# ── Step 3: Tokenize ─────────────────────────────────────────────────────────
inputs = tokenizer(passage, return_tensors="pt", truncation=True)

# ── Step 4: Generate summary ─────────────────────────────────────────────────
summary_ids = model.generate(
    inputs["input_ids"],
    max_length=80,
    min_length=25,
    do_sample=False
)

# ── Step 5: Decode and print ─────────────────────────────────────────────────
summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

print("ORIGINAL TEXT:")
print(passage.strip())

print("SUMMARIZED TEXT:")
print(summary)
