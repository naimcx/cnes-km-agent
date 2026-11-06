import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

# 1. Load environment variables from .env file
load_dotenv()

# 2. Instantiate the LLM
# temperature=0 ensures that the model's responses are consistent and reproducible.
llm = ChatGroq(
    model_name="openai/gpt-oss-120b",
    temperature=0
)

# 3. Basic test
reponse = llm.invoke("Hello, confirm that you are working correctly.")
print("Response from the model :", reponse.content)