from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import pandas as pd
import os

# === Detect if running on Streamlit Cloud ===
IS_CLOUD = os.getenv("STREAMLIT_SHARING_MODE") is not None

if IS_CLOUD:
    # Use Groq on Cloud
    from langchain_groq import ChatGroq
    llm = ChatGroq(
        model="llama3-70b-8192",   # Fast and strong
        temperature=0.3,
        api_key=os.getenv("GROQ_API_KEY")
    )
else:
    # Use local Ollama on your Mac
    from langchain_ollama import ChatOllama
    llm = ChatOllama(
        model="qwen2.5:14b",
        temperature=0.3,
        num_ctx=4096
    )

# Simple data loader (no vector DB needed on cloud)
def load_pricing_data():
    df = pd.read_csv("data/sample_pricing.csv")
    return df

pricing_df = load_pricing_data()

prompt = ChatPromptTemplate.from_template("""
You are an expert pricing strategist for an industrial supplier.

Product Data:
{context}

User Request: {question}

Give clear, professional recommendations with:
1. Suggested new price for each product
2. Expected margin improvement
3. Reasoning (cost, competition, demand)
4. Confidence level (High / Medium / Low)
""")

chain = prompt | llm | StrOutputParser()

def get_pricing_recommendation(question: str):
    context = pricing_df.to_string(index=False)
    return chain.invoke({"context": context, "question": question})