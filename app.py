"""
app.py — School AI Assistant

A Streamlit chat app that answers questions using a PRE-BUILT FAISS index.
It never re-embeds or re-processes the source PDFs at runtime — that work
is done once, offline, by ingest.py. This app only:
  1. Loads faiss_index/index.faiss + faiss_index/metadata.json
  2. Embeds the user's question (same embedding model used at ingest time)
  3. Retrieves the closest chunks (optionally restricted to one section)
  4. Sends those chunks + the question to Groq's openai/gpt-oss-120b model
  5. Streams back a grounded answer with source citations

Run with:
    streamlit run app.py

Requires a Streamlit secret:
    .streamlit/secrets.toml
        GROQ_API_KEY = "your-key-here"
"""

import os
import json

import numpy as np
import streamlit as st
import faiss
from sentence_transformers import SentenceTransformer
from groq import Groq


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
INDEX_DIR = "faiss_index"
INDEX_PATH = os.path.join(INDEX_DIR, "index.faiss")
METADATA_PATH = os.path.join(INDEX_DIR, "metadata.json")

EMBED_MODEL_NAME = "all-MiniLM-L6-v2"   # must match the model used in ingest.py
GROQ_MODEL = "openai/gpt-oss-120b"

TOP_K = 5              # chunks to feed the LLM after filtering
FETCH_K = 40            # chunks pulled from FAISS before section filtering

SCHOOL_NAME = "School AI Assistant"
ACCENT_COLOR = "#1F4E79"
ACCENT_LIGHT = "#EAF1F8"


# --------------------------------------------------------------------------
# Branding / CSS
# --------------------------------------------------------------------------
def inject_branding():
    st.markdown(
        f"""
        <style>
            /* Main background */
            .stApp {{
                background-color: #FAFBFC;
                color: #111827 !important;
            }}
            
            /* Fix invisible text in chat messages */
            .stChatMessage, .stChatMessage p, .stChatMessage div, [data-testid="stChatMessageContent"] {{
                color: #111827 !important;
            }}

            /* Clear background for chat containers */
            [data-testid="stChatMessage"] {{
                background-color: #F3F4F6 !important;
                border-radius: 10px;
            }}
            
            /* Sidebar Styling - Dark Theme */
            [data-testid="stSidebar"] {{
                background-color: #111827 !important;
                border-right: 1px solid #1F2937 !important;
            }}
            
            /* Sidebar Text & Labels */
            [data-testid="stSidebar"] h1, 
            [data-testid="stSidebar"] h2, 
            [data-testid="stSidebar"] h3, 
            [data-testid="stSidebar"] label, 
            [data-testid="stSidebar"] span {{
                color: #FFFFFF !important;
            }}
            
            /* Sidebar Captions */
            [data-testid="stSidebar"] [data-testid="stCaptionContainer"],
            [data-testid="stSidebar"] p {{
                color: #9CA3AF !important;
            }}

            /* Header Styling */
            .app-header {{
                display: flex;
                align-items: center;
                gap: 0.75rem;
                padding: 1rem 1.25rem;
                background-color: {ACCENT_COLOR};
                border-radius: 10px;
                margin-bottom: 1.25rem;
            }}
            .app-header h1 {{
                color: white !important;
                font-size: 1.4rem;
                margin: 0;
                font-weight: 600;
            }}
            .app-header p {{
                color: #DCE8F5 !important;
                margin: 0;
                font-size: 0.85rem;
            }}
            .source-pill {{
                display: inline-block;
                background-color: {ACCENT_LIGHT};
                color: {ACCENT_COLOR};
                border: 1px solid #B9D0E6;
                border-radius: 999px;
                padding: 2px 10px;
                margin: 2px 4px 0 0;
                font-size: 0.75rem;
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class="app-header">
            <div>🎓</div>
            <div>
                <h1>{SCHOOL_NAME}</h1>
                <p>Ask about grading, attendance, and other school policies</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------
# Cached resource loaders — run once per session, never re-process PDFs
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading knowledge base...")
def load_index_and_metadata():
    if not os.path.exists(INDEX_PATH) or not os.path.exists(METADATA_PATH):
        return None, None

    index = faiss.read_index(INDEX_PATH)
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    return index, metadata


@st.cache_resource(show_spinner="Loading embedding model...")
def load_embedder():
    return SentenceTransformer(EMBED_MODEL_NAME)


@st.cache_resource(show_spinner=False)
def load_groq_client():
    api_key = st.secrets.get("GROQ_API_KEY")
    if not api_key:
        return None
    return Groq(api_key=api_key)


def section_label(source_file: str) -> str:
    """Turn a filename like 'grading_system.pdf' into 'Grading System'."""
    name = os.path.splitext(source_file)[0]
    name = name.replace("_", " ").replace("-", " ")
    return name.strip().title()


# --------------------------------------------------------------------------
# Retrieval + generation
# --------------------------------------------------------------------------
def retrieve_chunks(query: str, index, metadata, embedder, allowed_sources):
    query_vec = embedder.encode(
        [query], convert_to_numpy=True, normalize_embeddings=True
    ).astype("float32")

    k = min(FETCH_K, index.ntotal)
    if k == 0:
        return []

    scores, indices = index.search(query_vec, k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        item = metadata[idx]
        if allowed_sources is not None and item["source_file"] not in allowed_sources:
            continue
        results.append({**item, "score": float(score)})
        if len(results) >= TOP_K:
            break

    return results


def build_prompt(question: str, chunks: list) -> list:
    if not chunks:
        context_block = "No relevant context was found in the knowledge base."
    else:
        context_block = "\n\n".join(
            f"[Source: {c['source_file']}]\n{c['text']}" for c in chunks
        )

    system_prompt = (
        "You are the School AI Assistant. Answer the user's question using ONLY "
        "the context provided below, which comes from official school policy "
        "documents. If the context does not contain the answer, say you don't "
        "have that information in the knowledge base and suggest the user check "
        "with the school office — do not make anything up. Keep answers clear "
        "and concise, and mention which document(s) the answer comes from when "
        "relevant.\n\n"
        f"CONTEXT:\n{context_block}"
    )

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]


def generate_answer(client: Groq, messages: list):
    """Streams the answer back as a generator of text deltas."""
    stream = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=messages,
        temperature=0.2,
        stream=True,
    )
    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta


# --------------------------------------------------------------------------
# Main app
# --------------------------------------------------------------------------
def main():
    st.set_page_config(
        page_title=SCHOOL_NAME,
        page_icon="🎓",
        layout="centered",
    )
    inject_branding()

    index, metadata = load_index_and_metadata()

    if index is None:
        st.error(
            f"No prebuilt index found at `{INDEX_DIR}/`. "
            "Run `python ingest.py` first to build the FAISS index — "
            "this app does not process PDFs itself."
        )
        st.stop()

    embedder = load_embedder()
    client = load_groq_client()

    # ---- Sidebar: knowledge base sections ----
    all_sources = sorted({item["source_file"] for item in metadata})
    labels = {src: section_label(src) for src in all_sources}

    with st.sidebar:
        st.markdown("### 📚 Knowledge Base")
        st.caption("Restrict answers to one or more sections.")

        select_all = st.checkbox("All sections", value=True)

        selected_labels = st.multiselect(
            "Sections",
            options=[labels[src] for src in all_sources],
            default=[labels[src] for src in all_sources] if select_all else [],
            disabled=select_all,
            label_visibility="collapsed",
        )

        if select_all:
            allowed_sources = None  # no filtering
        else:
            reverse_labels = {v: k for k, v in labels.items()}
            allowed_sources = {reverse_labels[l] for l in selected_labels} or set()

        st.divider()
        st.caption(f"{len(all_sources)} document(s) indexed · {len(metadata)} chunks")

        if st.button("🗑️ Clear chat"):
            st.session_state.messages = []
            st.rerun()

    if client is None:
        st.warning(
            "No Groq API key found. Add `GROQ_API_KEY` to your Streamlit "
            "secrets (`.streamlit/secrets.toml` or the app's Secrets settings) "
            "to enable answers."
        )

    # ---- Chat state ----
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                pills = "".join(
                    f'<span class="source-pill">{labels.get(s, s)}</span>'
                    for s in msg["sources"]
                )
                st.markdown(pills, unsafe_allow_html=True)

    # ---- Chat input ----
    question = st.chat_input("Ask about attendance, grading, or any school policy...")

    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            if client is None:
                answer_text = (
                    "I can't generate an answer right now because no Groq API "
                    "key is configured for this app."
                )
                st.markdown(answer_text)
                sources_used = []
            elif allowed_sources is not None and len(allowed_sources) == 0:
                answer_text = (
                    "Please select at least one section in the sidebar, "
                    "or choose **All sections**, before asking a question."
                )
                st.markdown(answer_text)
                sources_used = []
            else:
                chunks = retrieve_chunks(
                    question, index, metadata, embedder, allowed_sources
                )
                messages = build_prompt(question, chunks)

                placeholder = st.empty()
                answer_text = ""
                for delta in generate_answer(client, messages):
                    answer_text += delta
                    placeholder.markdown(answer_text + "▌")
                placeholder.markdown(answer_text)

                sources_used = sorted({c["source_file"] for c in chunks})
                if sources_used:
                    pills = "".join(
                        f'<span class="source-pill">{labels.get(s, s)}</span>'
                        for s in sources_used
                    )
                    st.markdown(pills, unsafe_allow_html=True)

        st.session_state.messages.append(
            {"role": "assistant", "content": answer_text, "sources": sources_used}
        )


if __name__ == "__main__":
    main()
