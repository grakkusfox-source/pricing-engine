import streamlit as st
import pandas as pd
import plotly.express as px
from agent import get_pricing_recommendation

st.set_page_config(page_title="AI Pricing Engine", layout="wide")
st.title("🚀 AI Pricing Recommendation Engine")
st.markdown("**Personal AI Portfolio Project 1**")

# File uploader
uploaded_file = st.file_uploader("Upload your pricing data (CSV)", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
else:
    df = pd.read_csv("data/sample_pricing.csv")

st.subheader("Current Pricing Data")
st.dataframe(df, width="stretch")

# Charts
col1, col2 = st.columns(2)
with col1:
    fig = px.bar(df, x="product_name", y="historical_sales", title="Historical Sales Volume")
    st.plotly_chart(fig, use_container_width=True)
with col2:
    fig2 = px.scatter(df, x="base_cost", y="competitor_price", 
                     size="historical_sales", color="product_name",
                     title="Cost vs Competitor Price")
    st.plotly_chart(fig2, use_container_width=True)

# AI Recommendations
st.subheader("Ask AI for Pricing Recommendations")
question = st.text_area(
    "Your request to the AI:",
    "Recommend dynamic prices for next quarter aiming for 8-12% margin improvement while staying competitive.",
    height=100
)

if st.button("Get AI Recommendations", type="primary"):
    with st.spinner("AI is analyzing..."):
        response = get_pricing_recommendation(question)
        st.success("✅ AI Recommendation")
        st.markdown(response)

st.caption("Built locally with Qwen2.5 14B + LangChain + Streamlit")