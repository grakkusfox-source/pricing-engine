# AI Pricing Recommendation Engine

**Personal AI Portfolio Project 1**

A fully local GenAI-powered pricing optimization tool that analyzes historical sales and pricing data to generate intelligent dynamic pricing recommendations.

### Key Features
- Upload your own pricing CSV files
- Interactive charts (sales volume + cost vs competitor analysis)
- AI agent with Retrieval-Augmented Generation (RAG)
- Fully offline — runs 100% locally on Mac Mini
- Zero ongoing API costs

### Tech Stack
- **LLM**: Qwen2.5 14B (via Ollama)
- **Orchestration**: LangChain
- **Vector Database**: Chroma
- **Frontend**: Streamlit
- **Visualizations**: Plotly

### How to Run Locally

```bash
cd pricing-engine
source .venv/bin/activate
streamlit run app.py
