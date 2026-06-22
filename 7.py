#pip -q install transformers==4.40.2 torch
#run in python 3.10 or 3.11, not in 3.12 because of some issues with transformers library, also set venv and install 
from transformers import pipeline

# Load model
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# Input text
text = """The Industrial Revolution, which took place from the 18th to the 19th centuries, 
was a period during which predominantly agrarian, 
rural societies in Europe and America became industrial and urban.


Prior to the Industrial Revolution, manufacturing was often 
done in people’s homes, using hand tools or basic machines. 
Industrialization marked a shift to powered, special-purpose machinery, factories and mass production. 

The iron and textile industries, along with the development of the steam engine, 
played central roles in the Industrial Revolution,which also saw improved systems of transportation, communication and banking. 

While industrialization brought about an increased 
volume and variety of manufactured goods and an improved standard of living for some, 
it also resulted in often grim employment and living conditions 
for the poor and working classes."""

print("Original:\n", text)

# 4 variants, s1 is default, s2 is more creative, s3 is more structured, s4 is more diverse
#s1 is enough for the task, but the others are just to show how different parameters work
s1 = summarizer(text, max_length=50, min_length=20, do_sample=False)
s2 = summarizer(text, max_length=50, min_length=20, do_sample=True)
s3 = summarizer(text, max_length=50, min_length=20, do_sample=False, num_beams=5)
s4 = summarizer(text, max_length=50, min_length=20, do_sample=True, top_k=50, top_p=0.9)

# Output
print("\nDefault:\n", s1[0]['summary_text'])
print("\nCreative:\n", s2[0]['summary_text'])
print("\nStructured:\n", s3[0]['summary_text'])
print("\nDiverse:\n", s4[0]['summary_text'])