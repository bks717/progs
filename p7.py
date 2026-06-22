%pip install transformers==4.40.2 torch
from transformers import pipeline

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

def summarize(text):
    text = " ".join(text.split())
    return summarizer("summarize: " + text, max_length=120, min_length=30)[0]["summary_text"]

long_text = '''Artificial Intelligence is a rapidly evolving field of computer science focused on
              creating intelligent machines capable of mimicking human cognitive functions such as learning, problem-solving, 
              and decision-making. In recent years, AI has significantly impacted various
              industries, including healthcare, finance, education, and entertainment. AI-powered
              applications, such as chatbots, self-driving cars, and recommendation systems, have
              transformed the way we interact with technology.'''

print("\nOriginal Text:\n", long_text)
print("\nSummary:\n", summarize(long_text))
