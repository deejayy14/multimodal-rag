# Multimodal RAG Assistant (Text - Image - Audio)

A **multimodal Retrieval-Augmented Generation (RAG)** system that supports **text, documents(with images) and voice input** built using **LangChain**, **Pinecone** and **self-hosted Ollama LLMs** with a **Streamlit** frontend.
This project demonstrates how to build and deploy a **RAG system with conversational memory**, multimodal reasoning and local LLM inference.

Live Demo:
https://witnesses-mat-clearance-hereby.trycloudflare.com/
This app uses self hosted LLMs, which requires a local runtime.
If the link is inactive, please refer to the local setup instructions below.


## Key Features:
-  **RAG with Pinecone** (document ingestion + semantic retrieval)
-  **Persistent conversational memory** (follow up questions work correctly)
-  **Image understanding**
  - OCR (text from images)
  - Image captioning (via vision LLM)
-  **Speech-to-Text** (AssemblyAI)
-  **Text-to-Speech** (gttS)
-  **PDF, text and image ingestion**
-  **Self hosted LLM inference** using Ollama
-  **Public demo via Cloudflare Tunnel**

## Architecture Overview:
User -> Streamlit UI -> LangChain [Query Rewriting,Conversational Memory] -> Pinecone [Vector Retrieval] -> Ollama LLMs (Local) [Text LLM (Qwen), Vision LLM (Qwen-VL)]

## Technical Highlights:
- Persistent chat history 
- Query rewriting for context aware follow up questions
- Multimodal retrieval (text + image)
- Clean separation of ingestion,retrieval and generation logic

## Author:
Dhananjay,
B.Tech CSE (AI),
Aspiring ML/LLM Engineer.


### Setup Instructions:

```bash
#install ollama
ollama pull qwen2.5:3b
ollama pull qwen3-vl:2b
git clone https://github.com/<your-username>/multimodal-rag.git
cd multimodal-rag
#create and activate virtual env
pip install -r requirements.txt
streamlit run app.py
[Create a .env file:
ASSEMBLYAI_API_KEY=your_key_here
PINECONE_API_KEY=your_key_here]


