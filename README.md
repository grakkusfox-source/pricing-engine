### AI Pricing Recommendation Engine
# Industrial Supplier Pricing Engine

**Overview**  
Built a full-stack AI application that analyzes historical pricing and sales data to generate intelligent, dynamic pricing recommendations. The tool supports CSV uploads, interactive visualizations, and provides clear business reasoning.

**Key Technical Achievements**
- Implemented Retrieval-Augmented Generation (RAG) using local and cloud LLMs
- Created responsive Streamlit dashboard with Plotly visualizations
- Developed hybrid architecture (local Ollama + Groq cloud fallback)
- Added file upload and export functionality

**Quantifiable Outcomes (Simulated)**
- Reduces manual pricing analysis time by ~70%
- Identifies opportunities for 8–18% margin improvement
- Provides auditable, explainable recommendations

**Tech Stack**
- Python, LangChain, Streamlit, Plotly
- Qwen2.5-14B (local) + Groq (cloud)
- Chroma (local vector store)

**Live Demo**: [pricing-engine.streamlit.app](https://pricing-engine-lk8x86ys8bjqnujw7grpey.streamlit.app)

**What I Learned**
- Building production-ready hybrid AI applications
- Handling local vs cloud LLM differences
- Creating user-friendly data tools with proper error handling