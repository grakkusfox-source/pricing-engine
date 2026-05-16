from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import pandas as pd
import os

# === Smart Model Selection ===
if os.getenv("STREAMLIT_SHARING_MODE"):   # Running on Streamlit Cloud
    from langchain_groq import ChatGroq
    llm = ChatGroq(
        model="qwen2.5-14b",           # or "llama3-70b-8192"
        temperature=0.3,
        api_key=os.getenv("GROQ_API_KEY")   # We'll set this in Streamlit
    )
    embeddings = OllamaEmbeddings(model="nomic-embed-text")  # Keep local embeddings for now
else:
    # Local Mac Mini
    llm = ChatOllama(model="qwen2.5:14b", temperature=0.3, num_ctx=4096)
    embeddings = OllamaEmbeddings(model="nomic-embed-text")

def load_pricing_data():
    df = pd.read_csv("data/sample_pricing.csv")
    texts = df.apply(lambda row: f"Product: {row['product_name']} ({row['category']}) | Base Cost: ${row['base_cost']} | Competitor: ${row['competitor_price']} | Past Sales: {row['historical_sales']} | Target Margin: {row['margin_target']}", axis=1).tolist()
    
    vectorstore = Chroma.from_texts(texts, embeddings, persist_directory="./chroma_db")
    return vectorstore.as_retriever(search_kwargs={"k": 4}), df

retriever, pricing_df = load_pricing_data()

prompt = ChatPromptTemplate.from_template("""
You are an expert pricing strategist.

Context: {context}

User Request: {question}

Provide clear recommendations including:
1. Suggested new price for each product
2. Expected margin improvement
3. Reasoning based on cost, competition, and demand
4. Confidence level (High / Medium / Low)
""")

chain = prompt | llm | StrOutputParser()

def get_pricing_recommendation(question: str):
    docs = retriever.invoke(question)
    context = "\n".join([doc.page_content for doc in docs])
    return chain.invoke({"context": context, "question": question})