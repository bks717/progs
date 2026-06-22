#pip install langchain langchain-community langchain-cohere cohere
from langchain_core.prompts import PromptTemplate
from langchain_cohere import ChatCohere
import getpass

# Load text file
text = open("sample.txt", "r", encoding="utf-8").read()

# API Key
api_key = getpass.getpass("Enter Cohere API Key: ")

# Model
llm = ChatCohere(
    cohere_api_key=api_key,
    model="command-r-08-2024"
)

# Prompt Template
prompt = PromptTemplate.from_template("""
Summarize the text and give key points.

Text:
{text}

Output:

Summary:
Key Points:
Sentiment:
""")

# Run
response = llm.invoke(prompt.format(text=text))

print(response.content)