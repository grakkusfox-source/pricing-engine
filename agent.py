import os
import pandas as pd
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# ================== ENVIRONMENT DETECTION ==================
IS_CLOUD = os.getenv("STREAMLIT_SHARING_MODE") is not None

if IS_CLOUD:
    # === Running on Streamlit Cloud ===
    from langchain_groq import ChatGroq
    llm = ChatGroq(
        model="llama3-70b-8192",      # Strong and fast
        temperature=0.3,
        api_key=os.getenv("GROQ_API_KEY")
    )
else:
    # === Running locally on your Mac ===
    from langchain_ollama import ChatOllama
    llm = ChatOllama(
        model="qwen2.5:14b",
        temperature=0.3,
        num_ctx=4096
    )

# Load data (simple version - works everywhere)
def load_pricing_data():
    df = pd.read_csv("data/sample_pricing.csv")
    return df

pricing_df = load_pricing_data()

# Prompt
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
    
    # Debug line - remove later
    print("DEBUG: Running on CLOUD =", IS_CLOUD)
    
    return chain.invoke({"context": context, "question": question})