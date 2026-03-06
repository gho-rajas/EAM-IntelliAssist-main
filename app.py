import os
import re
import json
import base64
import logging
from typing import Dict, List, Tuple

import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

import ollama
import hashlib

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader

logging.basicConfig(level=logging.INFO)

# =========================
# CONFIG (same as your base)
# =========================
#EMBED_MODEL = "nomic-embed-text"
EMBED_MODEL = "BAAI/bge-base-en-v1.5"
# DATA_ROOT = "./Data"
PERSIST_DIR = "./chroma_hxgn"
os.makedirs(PERSIST_DIR, exist_ok=True)

WORK_TYPES = [
    "Work Request",
    "Preventive Maintenance",
    "Asset Master Data",
    "Materials & Inventory",
    "Procurement",
    "Permits & Safety",
    "Mobile",
    "Reporting & Analytics",
    "Security & Roles",
    "Integrations"
]

# =========================
# Azure GPT from ENV (same as your base)
# =========================
load_dotenv()
AZURE_OPENAI_ENDPOINT = os.getenv("endpoint")
AZURE_OPENAI_API_KEY = os.getenv("api_key")
AZURE_DEPLOYMENT_NAME = os.getenv("deployment_name")

if not AZURE_OPENAI_ENDPOINT or not AZURE_OPENAI_API_KEY or not AZURE_DEPLOYMENT_NAME:
    raise RuntimeError("Missing env vars: endpoint, api_key, deployment_name (set in .env or environment).")

client = OpenAI(
    base_url=AZURE_OPENAI_ENDPOINT,
    api_key=AZURE_OPENAI_API_KEY,
)

def azure_chat(messages, deployment_name: str = AZURE_DEPLOYMENT_NAME, temperature: float = 0.2) -> str:
    resp = client.chat.completions.create(
        model=deployment_name,
        messages=messages,
        temperature=temperature,
    )
    return resp.choices[0].message.content


# =========================
# BASE FUNCTIONS (keep logic)
# =========================
def safe_collection_name(prefix: str, work_type: str) -> str:
    name = f"{prefix}_{work_type.lower()}"
    name = re.sub(r"[^a-z0-9._-]+", "_", name)
    name = re.sub(r"_{2,}", "_", name)
    name = re.sub(r"^[^a-z0-9]+", "", name)
    name = re.sub(r"[^a-z0-9]+$", "", name)
    if len(name) < 3:
        name = f"{prefix}_db"
    return name[:512]

def split_documents(documents: list, chunk_size: int = 1200, chunk_overlap: int = 300) -> list:
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_documents(documents)
    return chunks

# def build_or_load_vector_db(work_type: str) -> Chroma:
#     ollama.pull(EMBED_MODEL)

#     collection_name = safe_collection_name("hxgn", work_type)
#     persist_directory = os.path.join(PERSIST_DIR, safe_collection_name("dir", work_type))
#     os.makedirs(persist_directory, exist_ok=True)

#     embeddings = OllamaEmbeddings(model=EMBED_MODEL)

#     return Chroma(
#         collection_name=collection_name,
#         persist_directory=persist_directory,
#         embedding_function=embeddings,
#     )

from langchain_community.embeddings import SentenceTransformerEmbeddings
def build_or_load_vector_db(work_type: str) -> Chroma:
    collection_name = safe_collection_name("hxgn", work_type)
    persist_directory = os.path.join(PERSIST_DIR, safe_collection_name("dir", work_type))
    os.makedirs(persist_directory, exist_ok=True)

    embeddings = SentenceTransformerEmbeddings(
        model_name=EMBED_MODEL
    )

    return Chroma(
        collection_name=collection_name,
        persist_directory=persist_directory,
        embedding_function=embeddings,
    )

def expand_queries(question: str, n: int = 3) -> list[str]:
    prompt = f"""Generate {n} different versions of the user's question to retrieve relevant documents from a Hexagon EAM documentation vector database.
Return each variant on a new line (no numbering, no bullets).

Original question: {question}
"""
    text = azure_chat([{"role": "user", "content": prompt}], temperature=0.0)
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    variants = []
    for l in lines:
        if l not in variants:
            variants.append(l)
        if len(variants) >= n:
            break
    if question not in variants:
        variants.insert(0, question)
    return variants[:n]

def dedupe_docs(docs):
    seen = set()
    out = []
    for d in docs:
        key = hashlib.md5((d.page_content + str(d.metadata)).encode("utf-8")).hexdigest()
        if key not in seen:
            seen.add(key)
            out.append(d)
    return out

def retrieve_with_multiquery(vectordb, question: str, k_per_query: int = 6, max_docs: int = 10):
    variants = expand_queries(question, n=3)

    all_docs = []
    for q in variants:
        all_docs.extend(vectordb.similarity_search(q, k=k_per_query))

    all_docs = dedupe_docs(all_docs)
    return all_docs[:max_docs], variants

AGENT_SYSTEM = """You are the Hexagon EAM Implementation Advisor Agent.
Use ONLY the provided CONTEXT for factual claims. If context is missing, say what is missing and what to check next.

Two help modes:
- training: If user has asked specific qustion, answer is less than 50 words. Else teach step-by-step + why it matters + tips to avoid common mistakes.
- next_step: act like EAM test coach: immediate next steps, validations, expected outcome, what to do if validation fails.
"""

def format_context_and_sources(docs):
    blocks = []
    sources = []
    for i, d in enumerate(docs, start=1):
        label = f"S{i}"
        src = d.metadata.get("source", "unknown")
        page = d.metadata.get("page", None)
        wt = d.metadata.get("work_type", None)
        blocks.append(f"[{label}] {d.page_content}")
        sources.append({"label": label, "source": src, "page": page, "work_type": wt})
    return "\n\n".join(blocks), sources

def single_agent_answer(intake: dict, VECTOR_DBS: Dict[str, Chroma]) -> dict:
    required = ["objective","type_of_work","help_type","problem"]
    missing = [k for k in required if not intake.get(k)]
    if missing:
        return {"answer": f"Missing required fields: {missing}", "sources": []}

    work_type = intake["type_of_work"]
    help_type = intake["help_type"].strip().lower()
    if help_type not in ["training","next_step"]:
        return {"answer": "help_type must be 'training' or 'next_step'", "sources": []}

    vectordb = VECTOR_DBS[work_type]

    retrieval_question = f"""
Objective: {intake['objective']}
Problem: {intake['problem']}"""

    docs, variants = retrieve_with_multiquery(vectordb, retrieval_question, k_per_query=6, max_docs=10)
    context, sources = format_context_and_sources(docs)

    user_prompt = f"""HELP MODE: {help_type}

USER INTAKE:
{intake}

QUERY VARIANTS USED FOR RETRIEVAL:
{variants}

CONTEXT (cite sources like [S1], [S2] inline):
{context}
"""

    answer = azure_chat(
        [
            {"role": "system", "content": AGENT_SYSTEM},
            {"role": "user", "content": user_prompt},
        ],
        deployment_name=AZURE_DEPLOYMENT_NAME,
        temperature=0.2,
    )
    return {"answer": answer, "sources": sources}

# =========================
# Vision (images) — optional
# =========================
def _image_to_data_url(image_bytes: bytes, mime: str) -> str:
    b64 = base64.b64encode(image_bytes).decode("utf-8")
    return f"data:{mime};base64,{b64}"

def azure_chat_with_images(system_prompt: str, user_text: str, image_files: List[Tuple[str, bytes, str]], temperature: float = 0.2) -> str:
    content_parts = [{"type": "text", "text": user_text}]
    for filename, bts, mime in image_files:
        content_parts.append({
            "type": "image_url",
            "image_url": {"url": _image_to_data_url(bts, mime), "detail": "auto"}
        })

    resp = client.chat.completions.create(
        model=AZURE_DEPLOYMENT_NAME,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content_parts},
        ],
        temperature=temperature,
    )
    return resp.choices[0].message.content

def single_agent_answer_with_images(intake: dict, VECTOR_DBS: Dict[str, Chroma], image_files: List[Tuple[str, bytes, str]]) -> dict:
    # keep same logic; only final call becomes multimodal
    required = ["objective","type_of_work","help_type","problem"]
    missing = [k for k in required if not intake.get(k)]
    if missing:
        return {"answer": f"Missing required fields: {missing}", "sources": []}

    work_type = intake["type_of_work"]
    help_type = intake["help_type"].strip().lower()
    if help_type not in ["training","next_step"]:
        return {"answer": "help_type must be 'training' or 'next_step'", "sources": []}

    vectordb = VECTOR_DBS[work_type]

    retrieval_question = f"""
Objective: {intake['objective']}
Problem: {intake['problem']}"""

    docs, variants = retrieve_with_multiquery(vectordb, retrieval_question, k_per_query=6, max_docs=10)
    context, sources = format_context_and_sources(docs)

    user_prompt = f"""HELP MODE: {help_type}

USER INTAKE:
{intake}

QUERY VARIANTS USED FOR RETRIEVAL:
{variants}

CONTEXT (cite sources like [S1], [S2] inline):
{context}
"""

    answer = azure_chat_with_images(
        system_prompt=AGENT_SYSTEM,
        user_text=user_prompt,
        image_files=image_files,
        temperature=0.2
    )
    return {"answer": answer, "sources": sources}

# =========================
# Append-only doc ingest (adds only new file chunks)
# =========================
def _load_docs_from_file(path: str, work_type: str):
    if path.lower().endswith(".pdf"):
        loaded = PyPDFLoader(path).load()
    elif path.lower().endswith(".docx"):
        loaded = Docx2txtLoader(path).load()
    else:
        return []
    for d in loaded:
        d.metadata["source"] = path
        d.metadata["work_type"] = work_type
    return loaded

def add_uploaded_files_to_db(work_type: str, vectordb: Chroma, uploads: List[st.runtime.uploaded_file_manager.UploadedFile],
                             chunk_size: int = 1200, chunk_overlap: int = 300) -> Dict[str, int]:
    """
    Saves files into ./Data/<work_type>/ and appends only those file chunks into the existing collection.
    """
    saved = 0
    added_chunks = 0

    folder = os.path.join(DATA_ROOT, work_type)
    os.makedirs(folder, exist_ok=True)

    for f in uploads:
        out_path = os.path.join(folder, f.name)
        with open(out_path, "wb") as wf:
            wf.write(f.getvalue())
        saved += 1

        docs = _load_docs_from_file(out_path, work_type)
        if docs:
            chunks = split_documents(docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            vectordb.add_documents(chunks)
            added_chunks += len(chunks)

    return {"files_saved": saved, "chunks_added": added_chunks}

# =========================
# STREAMLIT UI (Copilot-like, simplified)
# - Agent name: EAM IntelliAssist
# - Sidebar: only Settings (Type of work + Mode)
# - No navigation, no actions, no role/stage
# =========================
import streamlit as st
from typing import Dict, List, Tuple

st.set_page_config(
    page_title="EAM IntelliAssist",
    page_icon="🛠️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- Theme ----------
ACCENT = "#0B3D2E"   # dark green
BORDER = "#E7E8EA"
MUTED = "#6B7280"
BG = "#FFFFFF"
CARD_BG = "#FFFFFF"

CSS = f"""
<style>
/* ---------- Base page ---------- */
html, body, [data-testid="stAppViewContainer"] {{
  background: {BG};
}}

/* ---------- Sidebar ---------- */
section[data-testid="stSidebar"] {{
  border-right: 1px solid {BORDER};
}}
section[data-testid="stSidebar"] > div {{
  padding-top: 0.8rem;
}}

/* ---------- Main width ---------- */
.block-container {{
  max-width: 1200px;
  padding-top: 1.1rem;
  padding-bottom: 1.8rem;
}}

/* ---------- Center welcome ---------- */
.h-title {{
  font-size: 2.0rem;
  font-weight: 650;
  text-align: center;
  margin: 2.0rem 0 0.65rem 0;
}}
.h-muted {{
  color: {MUTED};
  text-align: center;
  margin: 0 0 1.0rem 0;
  font-size: 0.95rem;
}}

/* ---------- Input shell ---------- */
.input-shell {{
  border: 1px solid {BORDER};
  border-radius: 16px;
  background: #fff;
  padding: 10px 12px;
  box-shadow: 0 10px 22px rgba(0,0,0,0.05);
}}
.tool-row {{
  display: flex;
  gap: 10px;
  align-items: center;
  padding: 6px 2px 0 2px;
  color: {MUTED};
  font-size: 0.88rem;
}}
.tool-chip {{
  display: inline-flex;
  gap: 8px;
  align-items: center;
  padding: 5px 9px;
  border-radius: 999px;
  border: 1px solid {BORDER};
  background: #fff;
}}

/* ---------- Suggestions: smaller + not emphasized ---------- */
.sugg-wrap {{
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 8px;
  margin-top: 10px;
}}
.sugg-card {{
  border: 1px solid {BORDER};
  background: #fff;
  border-radius: 12px;
  padding: 8px 10px;
  min-height: 44px;
}}
.sugg-title {{
  font-weight: 600;
  font-size: 0.90rem;
  margin-bottom: 2px;
}}
.sugg-sub {{
  color: {MUTED};
  font-size: 0.82rem;
  line-height: 1.2;
}}

/* ---------- Buttons: smaller + fix green text visibility ---------- */
.stButton > button {{
  border-radius: 10px;
  padding: 0.33rem 0.60rem;
  font-size: 0.85rem;
  border: 1px solid {ACCENT}22;
  color: #111827 !important;
  background: #fff;
}}
.stButton > button[kind="primary"] {{
  background: {ACCENT} !important;
  border-color: {ACCENT} !important;
  color: #FFFFFF !important;
}}
.stButton > button[kind="primary"]:hover {{
  background: {ACCENT} !important;
  border-color: {ACCENT} !important;
  filter: brightness(0.96);
  color: #FFFFFF !important;
}}

/* ---------- Inputs ---------- */
textarea, input {{
  border-radius: 12px !important;
  color: #111827 !important;
}}
textarea::placeholder, input::placeholder {{
  color: {MUTED} !important;
}}
[data-baseweb="select"] > div {{
  border-radius: 12px;
}}

/* ---------- Chat ---------- */
[data-testid="stChatMessage"] {{
  border-radius: 14px;
}}

/* ---------- Uploaders: smaller/tighter ---------- */
[data-testid="stFileUploader"] > section {{
  border-radius: 12px;
  padding: 8px 10px !important;
}}
[data-testid="stFileUploader"] label {{
  font-size: 0.9rem !important;
}}
[data-testid="stFileUploader"] div {{
  padding-top: 2px;
  padding-bottom: 2px;
}}

/* ---------- Tools cards: equal size ---------- */
.tools-grid {{
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 12px;
  margin-top: 10px;
}}
.tool-card {{
  border: 1px solid {BORDER};
  border-radius: 12px;
  background: #fff;
  padding: 10px 12px;
  min-height: 220px; /* adjust shorter/taller if needed */
}}
.tool-card .tool-cap {{
  color: {MUTED};
  font-size: 0.86rem;
  font-weight: 600;
  margin-bottom: 8px;
}}
.tool-spacer {{
  height: 36px; /* keeps height equal vs docs card having Index button */
}}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)
# -------------------------
# Session state
# -------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "draft" not in st.session_state:
    st.session_state.draft = ""

# -------------------------
# Load vector stores
# -------------------------
@st.cache_resource
def load_vector_dbs() -> Dict[str, Chroma]:
    dbs = {}
    for wt in WORK_TYPES:
        dbs[wt] = build_or_load_vector_db(wt)
    return dbs

VECTOR_DBS = load_vector_dbs()

# -------------------------
# Sidebar: ONLY Settings
# -------------------------
with st.sidebar:
    st.markdown("### EAM IntelliAssist")
    st.markdown("<div style='color:#6B7280; font-size:0.92rem;'>Settings</div>", unsafe_allow_html=True)

    type_of_work = st.selectbox("Type of work", WORK_TYPES, index=0, key="type_of_work_select")
    help_type = st.radio("Mode", ["training", "next_step"], index=0, key="mode_radio")

# -------------------------
# Main: Copilot-like landing + input
# -------------------------
is_empty = len(st.session_state.messages) == 0

if is_empty:
    st.markdown(
        "<div style='text-align:center; color:#6B7280; font-size:0.92rem; font-weight:600; margin-top:0.6rem;'>"
        "EAM IntelliAssist"
        "</div>",
        unsafe_allow_html=True,
    )
    st.markdown("<div class='h-title'>EAM Intelliguide</div>", unsafe_allow_html=True)
    st.markdown("<div class='h-title'>Welcome, how can I help?</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='h-muted'>Message EAM IntelliAssist — upload docs/screenshots.</div>",
        unsafe_allow_html=True,
    )

# Center area
left_pad, center, right_pad = st.columns([0.12, 0.76, 0.12])
with center:
    # Big message box
    #st.markdown("<div class='input-shell'>", unsafe_allow_html=True)

    with st.form("copilot_input", clear_on_submit=False):
        prompt = st.text_area(
            "Message",
            value=st.session_state.draft,
            placeholder="Message EAM IntelliAssist",
            height=80 if is_empty else 110,
            label_visibility="collapsed",
        )

        b1, b2 = st.columns([0.82, 0.18], gap="small")
        with b1:
            st.markdown(
                "<div class='tool-row'>"
                "</div>",
                unsafe_allow_html=True,
            )
        with b2:
            submitted = st.form_submit_button("Send", type="primary", use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

    # Fixed Tools (always visible)
    st.markdown("<div style='margin-top:12px;'></div>", unsafe_allow_html=True)
    tools_left, tools_right = st.columns([1.15, 0.85], gap="large")

    with tools_left:
        st.markdown("**Tools: Knowledge base (PDF/DOCX)**")
        doc_uploads = st.file_uploader(
            "Upload docs to index",
            type=["pdf", "docx"],
            accept_multiple_files=True,
            key="doc_uploads",
            label_visibility="collapsed",
        )
        index_btn = st.button(
            "Index into selected work-type",
            type="primary",
            use_container_width=True,
            disabled=not doc_uploads,
        )

    with tools_right:
        st.markdown("**Tools: Screenshots (images)**")
        image_uploads = st.file_uploader(
            "Upload screenshots for this question",
            type=["png", "jpg", "jpeg", "webp", "gif"],
            accept_multiple_files=True,
            key="image_uploads",
            label_visibility="collapsed",
        )
        if image_uploads and len(image_uploads) > 10:
            st.warning("Max 10 images per question. Extra images will be ignored.")
            image_uploads = image_uploads[:10]

    if index_btn:
        with st.spinner("Indexing documents..."):
            stats = add_uploaded_files_to_db(
                work_type=type_of_work,
                vectordb=VECTOR_DBS[type_of_work],
                uploads=doc_uploads,
                chunk_size=1200,
                chunk_overlap=300,
            )
        st.success(f"Indexed {stats['files_saved']} file(s) • {stats['chunks_added']} chunks into “{type_of_work}”.")

    # Suggestions (only when empty)
    if is_empty:
        st.markdown(
            """
            <div class="sugg-wrap">
              <div class="sugg-card">
                <div class="sugg-title">Validate a Work Request flow</div>
                <div class="sugg-sub">Next steps </div>
              </div>
              <div class="sugg-card">
                <div class="sugg-title">Prepare UAT test steps</div>
                <div class="sugg-sub">Structured steps and validation points.</div>
              </div>
              <div class="sugg-card">
                <div class="sugg-title">Work Request</div>
                <div class="sugg-sub">Step-by-step guidance</div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        s1, s2, s3 = st.columns(3, gap="large")
        with s1:
            if st.button("Use this", use_container_width=True, key="sugg1"):
                st.session_state.draft = "I need next_step guidance to validate the Work Request flow. What should I do, what should I see, and what if it fails?"
                st.rerun()
        with s2:
            if st.button("Use this", use_container_width=True, key="sugg2"):
                st.session_state.draft = "Create UAT test steps for my scenario and tell me expected results + validations."
                st.rerun()
        with s3:
            if st.button("Use this", use_container_width=True, key="sugg3"):
                st.session_state.draft = "Teach me how to configure Preventive Maintenance in Hexagon EAM step-by-step, and the most common mistakes to avoid."
                st.rerun()

# -------------------------
# Chat history (below once conversation starts)
# -------------------------
if len(st.session_state.messages) > 0:
    st.divider()    
    st.markdown("### Conversation")

    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
            if m["role"] == "assistant" and m.get("sources"):
                with st.expander("Sources"):
                    st.json(m["sources"])

# -------------------------
# Send handler
# -------------------------
def _run_question(user_text: str):
    st.session_state.messages.append({"role": "user", "content": user_text})

    # intake remains compatible with your required fields
    intake = {
        "objective": "Hexagon EAM implementation support",
        "type_of_work": type_of_work,
        "help_type": help_type,
        "problem": user_text,
    }

    image_payload: List[Tuple[str, bytes, str]] = []
    image_uploads_local = st.session_state.get("image_uploads", None)
    if image_uploads_local:
        for f in image_uploads_local[:10]:
            image_payload.append((f.name, f.getvalue(), f.type or "image/png"))

    with st.spinner("Working..."):
        try:
            if image_payload:
                result = single_agent_answer_with_images(intake, VECTOR_DBS, image_payload)
            else:
                result = single_agent_answer(intake, VECTOR_DBS)

            answer_text = result.get("answer", "")
            sources = result.get("sources", [])
        except Exception as e:
            answer_text = f"Error: {e}"
            sources = []

    st.session_state.messages.append({"role": "assistant", "content": answer_text, "sources": sources})

# If submitted, run and clear draft
if "submitted" in locals() and submitted:
    if prompt and prompt.strip():
        st.session_state.draft = ""
        _run_question(prompt.strip())
        st.rerun()
