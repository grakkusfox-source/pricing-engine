from langchain_ollama import ChatOllama
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import pandas as pd

# Initialize local model
llm = ChatOllama(
    model="qwen2.5:14b-q5_K_M",
    temperature=0.3,
    num_ctx=8192
)

embeddings = OllamaEmbeddings(model="nomic-embed-text")

def load_pricing_data():
    df = pd.read_csv("data/sample_pricing.csv")
    texts = df.apply(lambda row: f"Product: {row['product_name']} ({row['category']}) | Base Cost: ${row['base_cost']} | Competitor: ${row['competitor_price']} | Past Sales: {row['historical_sales']} | Target Margin: {row['margin_target']}", axis=1).tolist()
    
    vectorstore = Chroma.from_texts(texts, embeddings, persist_directory="./chroma_db")
    return vectorstore.as_retriever(), df

retriever, pricing_df = load_pricing_data()

prompt = ChatPromptTemplate.from_template("""
You are an expert pricing strategist. Use the retrieved context and recommend 
new dynamic prices.

Context: {context}
User Request: {question}

Give clear recommendations with:
1. Suggested price
2. Expected margin
3. Reasoning
4. Confidence level
""")

chain = prompt | llm | StrOutputParser()

def get_pricing_recommendation(question: str):
    docs = retriever.invoke(question)
    context = "\n".join([doc.page_content for doc in docs])
    return chain.invoke({"context": context, "question": question})
