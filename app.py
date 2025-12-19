import os
import streamlit as st
import requests
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import base64
from PIL import Image
import tempfile
import uuid
import time
from dotenv import load_dotenv
import pytesseract, cv2
from audiorecorder import audiorecorder

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda, RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from pinecone import Pinecone, ServerlessSpec
from docling.document_converter import DocumentConverter
import pymupdf as fitz
import assemblyai as aai
from pydub import AudioSegment
from gtts import gTTS

load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
ASSEMBLYAI_API_KEY = os.getenv("ASSEMBLYAI_API_KEY")

missing_keys = []
if not PINECONE_API_KEY:
    missing_keys.append("PINECONE_API_KEY")
if not ASSEMBLYAI_API_KEY:
    missing_keys.append("ASSEMBLYAI_API_KEY")
if missing_keys:
    st.error("❌ Missing required API keys:")
    for key in missing_keys:
        st.error(f"   - {key}")
    st.info(
        "📌 Fix: create a `.env` file with the required keys, e.g.\n"
        "PINECONE_API_KEY=...\n"
        "ASSEMBLYAI_API_KEY=...\n"
    )
    st.stop()

TEXT_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
IMAGE_EMBED_MODEL = "sentence-transformers/clip-ViT-B-32"
TEXT_DIM = 384
IMAGE_DIM = 512
TEXT_INDEX_NAME = "multimodal-rag-text"
IMAGE_INDEX_NAME = "multimodal-rag-image"
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_TEXT_MODEL = "qwen2.5:3b"
OLLAMA_VL_MODEL = "qwen3-vl:2b"
PINECONE_CLOUD = "aws"
PINECONE_REGION = "us-east-1"
TOP_K = 5
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100

@st.cache_resource
def init_pinecone():
    if ASSEMBLYAI_API_KEY:
        aai.settings.api_key = ASSEMBLYAI_API_KEY
    
    if not PINECONE_API_KEY:
        raise RuntimeError("PINECONE_API_KEY is missing")
    pc = Pinecone(api_key=PINECONE_API_KEY)
    existing = [i.name for i in pc.list_indexes()]

    if TEXT_INDEX_NAME not in existing:
        pc.create_index(
            name=TEXT_INDEX_NAME,
            dimension=TEXT_DIM,
            metric="cosine",
            spec=ServerlessSpec(cloud=PINECONE_CLOUD, region=PINECONE_REGION)
            )

    if IMAGE_INDEX_NAME not in existing:
        pc.create_index(
            name=IMAGE_INDEX_NAME,
            dimension=IMAGE_DIM,
            metric="cosine",
            spec=ServerlessSpec(cloud=PINECONE_CLOUD, region=PINECONE_REGION)
            )

    time.sleep(10)
    return pc

@st.cache_resource
def init_embeddings():
    text_emb = HuggingFaceEmbeddings(
        model_name=TEXT_EMBED_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )
    return text_emb

@st.cache_resource
def init_text_splitter():
    return RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )

@st.cache_resource
def init_llm():
    return OllamaLLM(
        model=OLLAMA_TEXT_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=0
    )

pc = init_pinecone()
if pc is None:
    st.error("❌ Pinecone client not initialized")
    st.stop()
text_embeddings = init_embeddings()
text_splitter = init_text_splitter()
llm = init_llm()

from sentence_transformers import SentenceTransformer
image_embeddings = None

def get_image_embedder():
    global image_embeddings
    if image_embeddings is None:
        image_embeddings = SentenceTransformer(IMAGE_EMBED_MODEL)
    return image_embeddings

#Helper fns:

def create_session_id():
    return f"session_{uuid.uuid4().hex[:8]}"

def image_to_base64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()
def ocr_image(image_path: str) -> str:
    try:
        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        text = pytesseract.image_to_string(gray)
        return text.strip()
    except Exception:
        return ""
def caption_image_with_qwen(image_path: str) -> str:
    try:
        payload = {
            "model": OLLAMA_VL_MODEL,
            "prompt": "Describe this image in detail. Include any text, diagrams, labels, or structure.",
            "images": [image_to_base64(image_path)],
            "stream": False
        }

        r = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json=payload,
            timeout=180
        )

        return r.json().get("response", "").strip()
    except Exception:
        return ""

def reset_audio_state():
    st.session_state.submit_audio_now = False
    st.session_state.recording_started = False
    st.session_state.last_audio_len = 0

MESSAGE_STORE: Dict[str, ChatMessageHistory] = {}

#Audio transcription:
def save_recorded_audio(audio_segment) -> str:
    fd, path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    audio_segment.export(path, format="wav")
    return path

def transcribe_audio(path: str) -> str:
    if not ASSEMBLYAI_API_KEY:
        return ""
    try:
        transcript = aai.Transcriber().transcribe(path)
        return transcript.text or ""
    except Exception as e:
        st.error(f"Audio transcription error: {e}")
        return ""
    finally:
        try:
            os.remove(path)
        except:
            pass

# Docs ingestion:
def process_pdf(path: str):
    docs, images = [], []
    try:
        result = DocumentConverter().convert(path)
        docs.append(Document(
            page_content=result.document.export_to_markdown(),
            metadata={"source": os.path.basename(path)}
        ))
    except:
        pdf = fitz.open(path)
        text = "".join(p.get_text() for p in pdf)
        if text.strip():
            docs.append(Document(
                page_content=text,
                metadata={"source": os.path.basename(path)}
            ))

    pdf = fitz.open(path)
    temp_dir = tempfile.mkdtemp()
    for i, page in enumerate(pdf):
        for j, img in enumerate(page.get_images()):
            base = pdf.extract_image(img[0])
            img_path = os.path.join(temp_dir, f"p{i}_i{j}.png")
            with open(img_path, "wb") as f:
                f.write(base["image"])
            #ocr
            ocr_text = ocr_image(img_path)
            if ocr_text:
                docs.append(Document(
                    page_content=ocr_text,
                    metadata={
                        "source": "image_ocr",
                        "image_path": img_path
                    }
                ))
            #captioning
            caption = caption_image_with_qwen(img_path)
            if caption:
                docs.append(Document(
                    page_content=caption,
                    metadata={
                        "source": "image_caption",
                        "image_path": img_path
                    }
                ))
            images.append({"image_path": img_path})
    return docs, images

def process_text_file(path: str):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            text = f.read()
        return [Document(
            page_content=text,
            metadata={"source": os.path.basename(path)}
        )]
    except Exception as e:
        st.error(f"Text file error: {e}")
        return []

def process_image(path: str):
    docs = []
    #ocr
    ocr_text = ocr_image(path)
    if ocr_text:
        docs.append(Document(
            page_content=ocr_text,
            metadata={"source": "image_ocr", "image_path": path}
        ))
    #captioning
    caption = caption_image_with_qwen(path)
    if caption:
        docs.append(Document(
            page_content=caption,
            metadata={"source": "image_caption", "image_path": path}
        ))
    return {"image_path": path}, docs

def store_documents(docs, namespace):
    vs = PineconeVectorStore(
        index_name=TEXT_INDEX_NAME,
        embedding=text_embeddings,
        namespace=namespace
    )
    chunks = text_splitter.split_documents(docs)
    vs.add_documents(chunks)
    return len(chunks)

def store_images(images, namespace):
    index = pc.Index(IMAGE_INDEX_NAME)
    embedder = get_image_embedder()
    for img in images:
        try:
            vec = embedder.encode(Image.open(img["image_path"]).convert("RGB")).tolist()
            index.upsert(
                [(uuid.uuid4().hex, vec, {"type": "image", "path": img["image_path"]})],
                namespace=namespace
            )
        except Exception as e:
            st.warning(f"Image storage error: {e}")
    return len(images)

def retrieve_images(query, namespace, k=2):
    try:
        q_vec = get_image_embedder().encode(query).tolist()
        res = pc.Index(IMAGE_INDEX_NAME).query(
            vector=q_vec, top_k=k, namespace=namespace,
            filter={"type": {"$eq": "image"}}
        )
        return [m.metadata["path"] for m in res.matches if os.path.exists(m.metadata["path"])]
    except:
        return []

#RAG Chain:
contextualize_prompt = ChatPromptTemplate.from_messages([
    ("system", """Rewrite the user's message as a standalone query.
     Use chat history to resolve references, names, or context."""),
    ("placeholder", "{chat_history}"),
    ("human", "{input}")
])

rewrite_query = (
    contextualize_prompt
    | llm
    | StrOutputParser()
)

qa_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a friendly, professional and conversational multimodal RAG assistant with memory.
You must use the chat history to remember the facts shared by the user.
If the user asks about something they previously stated, answer using chat history even if no documents are retrieved.
You can:
- Hold natural conversations
- Acknowledge personal facts shared by the user
- Use retrieved documents when available
- If the user states a personal fact, acknowledge it naturally
Don't say "no context provided" for a normal conversation."""),
    ("placeholder", "{chat_history}"),
    ("human", "Context:\n{context}\n\nQuestion: {input}")
])

answer_chain = (
    qa_prompt
    | llm
    | StrOutputParser()
)

def build_rag(namespace: str):
    vs = PineconeVectorStore(
        index_name=TEXT_INDEX_NAME,
        embedding=text_embeddings,
        namespace=namespace
    )
    retriever = vs.as_retriever(search_kwargs={"k": TOP_K})

    #rewriting query
    rewrite_step = (
        RunnableLambda(lambda x: {
            "input": x["input"],
            "chat_history": x["chat_history"]
        })
        | rewrite_query
    )

    #retrieving docs:
    def retrieve_docs(x):
        docs = retriever.invoke(x["standalone_question"])
        context = "\n".join(d.page_content for d in docs)
        return {
            "context": context,
            "input": x["standalone_question"],
            "chat_history": x["chat_history"]
        }

    retrieve_step = RunnableLambda(retrieve_docs)

    #Full Langchain pipeline:
    rag_chain = (
        {
            "standalone_question": rewrite_step,
            "input": RunnableLambda(lambda x: x["input"]),
            "chat_history": RunnableLambda(lambda x: x["chat_history"]),
        }
        | retrieve_step
        | answer_chain
    )
    def get_session_history(session_id: str) -> ChatMessageHistory:
        if session_id not in MESSAGE_STORE:
            MESSAGE_STORE[session_id] = ChatMessageHistory()
        return MESSAGE_STORE[session_id]

    return RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
    )

def process_chat(msg, audio_file, session_id):
    """Process chat message and return response"""
    
    if "rag_chain" not in st.session_state or st.session_state.rag_chain is None:
        st.session_state.rag_chain = build_rag(session_id)
    
    if audio_file is not None:
        msg = transcribe_audio(audio_file)
        if not msg:
            return None, None, "Could not transcribe audio"
    
    if not msg or not msg.strip():
        return None, None, None
    
    images = retrieve_images(msg, session_id)
    
    if images:
        #Using vision model:
        payload = {
            "model": OLLAMA_VL_MODEL,
            "prompt": msg,
            "images": [image_to_base64(p) for p in images[:2]],
            "stream": False
        }
        try:
            r = requests.post(f"{OLLAMA_BASE_URL}/api/generate", json=payload, timeout=180)
            resp = r.json()
            answer = resp.get("response", "⚠️ Model did not return a response.")
        except Exception as e:
            answer = f"⚠️ Error: {str(e)}"
    else:
        #Using text RAG model
        try:
            answer = st.session_state.rag_chain.invoke(
                {"input": msg},
                config={"configurable": {"session_id": session_id}}
            )
            if not answer or not answer.strip():
                answer = "⚠️ The model did not return a response. Please try again."
        except Exception as e:
            answer = f"⚠️ Error: {str(e)}"
    
    #Generate TTS:
    audio_response = None
    if answer and answer.strip():
        try:
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
            gTTS(answer[:500]).save(tmp.name)
            audio_response = tmp.name
        except:
            pass
    
    return answer, images, audio_response

#Streamlit UI:

st.set_page_config(
    page_title="Multimodal RAG Chat",
    page_icon="🧠",
    layout="wide"
)

if "session_id" not in st.session_state:
    st.session_state.session_id = create_session_id()
if "messages" not in st.session_state:
    st.session_state.messages = []
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None
if "audio_recorder_key" not in st.session_state:
    st.session_state["audio_recorder_key"] = 0
if "audio_consumed" not in st.session_state:
    st.session_state["audio_consumed"] = False
if "submit_audio_now" not in st.session_state:
    st.session_state.submit_audio_now = False
if "audio_processing" not in st.session_state:
    st.session_state.audio_processing = False
if "recording_armed" not in st.session_state:
    st.session_state.recording_armed = False
if "last_audio_len" not in st.session_state:
    st.session_state.last_audio_len = 0
if "recording_started" not in st.session_state:
    st.session_state.recording_started = False
if "audio_recorder_key" not in st.session_state:
    st.session_state.audio_recorder_key = 0

with st.sidebar:
    st.title("Multimodal RAG Assistant")
    st.markdown("**Using CLIP & Qwen-VisionLM**")
    
    st.divider()
    
    st.subheader("Session")
    st.text_input("Session ID", value=st.session_state.session_id, disabled=True)
    
    if st.button("New Session"):
        st.session_state.session_id = create_session_id()
        st.session_state.messages = []
        st.session_state.rag_chain = None
        st.rerun()
    
    st.divider()
    
    st.subheader("📁 Upload Files")
    uploaded_files = st.file_uploader(
        "Upload PDFs, images, or text files",
        type=["pdf", "txt", "jpg", "jpeg", "png", "bmp"],
        accept_multiple_files=True
    )
    st.session_state.submit_audio_now = False
    st.session_state.recording_armed = False
    
    if st.button("Process Files") and uploaded_files:
        reset_audio_state()
        st.session_state.submit_audio_now = False
        st.session_state.recording_armed = False
        st.session_state.last_audio_len = 0
        with st.spinner("Processing files..."):
            total_chunks = 0
            total_images = 0
            
            for uploaded_file in uploaded_files:
                temp_path = os.path.join(tempfile.gettempdir(), uploaded_file.name)
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                ext = Path(uploaded_file.name).suffix.lower()
                
                if ext == '.pdf':
                    docs, imgs = process_pdf(temp_path)
                    total_chunks += store_documents(docs, st.session_state.session_id)
                    total_images += store_images(imgs, st.session_state.session_id)
                
                elif ext in ['.jpg', '.jpeg', '.png', '.bmp']:
                    img_data, img_docs = process_image(temp_path)
                    if img_docs:
                        total_chunks += store_documents(img_docs, st.session_state.session_id)
                    total_images += store_images([img_data], st.session_state.session_id)
                
                elif ext == '.txt':
                    docs = process_text_file(temp_path)
                    total_chunks += store_documents(docs, st.session_state.session_id)
            
            st.success(f"✅ Uploaded: {total_chunks} text chunks, {total_images} images")
            
            st.session_state.rag_chain = None
    
    st.divider()
    
st.title("💬 Chat")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        if "images" in message and message["images"]:
            st.image(message["images"], width=200, caption="Retrieved Evidence")        
        if "audio" in message and message["audio"]:
            st.audio(message["audio"])

MIN_AUDIO_DURATION_MS = 1000
with st.sidebar:
    st.subheader("🎤 Audio Input")

    audio_input = st.file_uploader(
        "Upload audio file",
        type=["wav", "mp3", "m4a"],
        key="audio_upload"
    )

    recorded_audio = audiorecorder(
        "🎙️ Record",
        "🛑 Stop",
        key=f"audio_record_{st.session_state.audio_recorder_key}"
    )

    current_len = len(recorded_audio) if recorded_audio else 0
    prev_len = st.session_state.last_audio_len

    if current_len > 0 and prev_len == 0:
        st.session_state.recording_started = True
    if (
        st.session_state.recording_started
        and current_len > MIN_AUDIO_DURATION_MS
        and prev_len <= MIN_AUDIO_DURATION_MS
    ):
        st.session_state.recorded_audio_data = recorded_audio
        st.session_state.submit_audio_now = True
        st.session_state.recording_started = False  

    st.session_state.last_audio_len = current_len

    st.divider()

    if st.button("Clear Chat"):
        reset_audio_state()
        st.session_state.messages = []
        st.session_state.rag_chain = None
        st.rerun()
    
    st.divider()
    
    if st.button("Get Stats"):
        reset_audio_state()
        try:
            index = pc.Index(TEXT_INDEX_NAME)
            img_index = pc.Index(IMAGE_INDEX_NAME)
            stats = index.describe_index_stats()
            ns_stats = stats.namespaces.get(st.session_state.session_id, {})
            count = ns_stats.get('vector_count', 0)
            st.info(f"Vectors in namespace: {count}")
        except Exception as e:
            st.error(f"Error: {e}")

prompt = st.chat_input(
    "Ask anything...",
    disabled=st.session_state.audio_processing
    )

if prompt and not st.session_state.get("recorded_audio_active", False):
    st.session_state.messages.append({
        "role": "user",
        "content": prompt
    })

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer, images, audio_response = process_chat(
                prompt,
                None,
                st.session_state.session_id
            )

        if answer:
            st.markdown(answer)

            if images:
                st.image(images, width=200, caption="Retrieved Evidence")

            if audio_response:
                st.audio(audio_response)

            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "images": images,
                "audio": audio_response
            })

if st.session_state.submit_audio_now:
    st.session_state.submit_audio_now = False  
    st.session_state.audio_processing = True

    with st.chat_message("user"):
        st.markdown("🎙️ *Voice message*")

    with st.chat_message("assistant"):
        with st.spinner("Processing voice input..."):
            audio_path = save_recorded_audio(
                st.session_state.recorded_audio_data
            )

            answer, images, audio_response = process_chat(
                "",
                audio_path,
                st.session_state.session_id
            )

        if answer:
            st.markdown(answer)
            if images:
                st.image(images, width=200, caption="Retrieved Evidence")
            if audio_response:
                st.audio(audio_response)
            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "images": images,
                "audio": audio_response
            })

    st.session_state.pop("recorded_audio_data", None)
    st.session_state.submit_audio_now = False
    st.session_state.recording_armed = False
    st.session_state.last_audio_len = 0
    st.session_state.audio_processing = False
    st.session_state.audio_recorder_key += 1

