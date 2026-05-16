from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import pandas as pd
import os

# Cloud vs Local detection
if os.getenv("STREAMLIT_SHARING_MODE"):
    # Running on Streamlit Cloud → Use Groq (fast & free tier available)
    from langchain_groq import ChatGroq
    llm = ChatGroq(
        model="llama3-70b-8192",      # Very good and fast
        temperature=0.3,
        api_key=os.getenv("GROQ_API_KEY")
    )
else:
    # Running locally on your Mac
    from langchain_ollama import ChatOllama
    llm = ChatOllama(model="qwen2.5:14b", temperature=0.3, num_ctx=4096)

# Simple in-memory data (no Chroma on cloud)
def load_pricing_data():
    df = pd.read_csv("data/sample_pricing.csv")
    return df

pricing_df = load_pricing_data()

prompt = ChatPromptTemplate.from_template("""
You are an expert pricing strategist.

Use the following product data:

{context}

User Request: {question}

Give clear recommendations:
1. Suggested new price for each product
2. Expected margin improvement
3. Reasoning (cost, competition, demand)
4. Confidence level
""")

chain = prompt | llm | StrOutputParser()

def get_pricing_recommendation(question: str):
    # Simple context (no vector DB needed on cloud)
    context = pricing_df.to_string(index=False)
    return chain.invoke({"context": context, "question": question})