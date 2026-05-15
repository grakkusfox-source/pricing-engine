import streamlit as st
import pandas as pd
from agent import get_pricing_recommendation, pricing_df

st.set_page_config(page_title="Howden AI Pricing Engine", layout="wide")
st.title("🚀 AI Pricing Recommendation Engine")
st.markdown("**Project 1 - Howden Portfolio**")

st.subheader("Historical Pricing Data")
st.dataframe(pricing_df, use_container_width=True)

st.subheader("Ask the AI for Pricing Recommendations")
question = st.text_area(
    "What would you like to know?",
    "Recommend dynamic prices for next quarter that improve margins by 8-12% while staying competitive.",
    height=100
)

if st.button("Get AI Recommendations"):
    with st.spinner("AI is thinking..."):
        response = get_pricing_recommendation(question)
        st.success("✅ Recommendation Ready")
        st.markdown(response)

st.caption("Built locally with Qwen2.5 14B • Zero API cost")
