#!/usr/bin/env python3

"""
Streamlit app for RAG-based Q&A about human resources topics. The app allows users to input questions, which are sent to a FastAPI backend that performs retrieval from a vector database and generates answers with grounding information.

Usage:

    1. Start the FastAPI backend (app/api.py)

    > uvicorn app.api:app --reload --host 0.0.0.0 --port 8000

    2. Run this Streamlit app

    > streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import logging
from pathlib import Path

import requests
import streamlit as st
from dotenv import load_dotenv

from rrhh_rag import conf

logger = logging.getLogger(__name__)

load_dotenv()

# Configuration
try:
    conf_settings = conf.load(file="settings.yaml")
except Exception as e:
    logger.error("Failed to load conf files: %s", e)

API_URL = conf_settings.rag_api_url

BASE_DIR = Path(__file__).resolve().parent
LOGO_PATH = BASE_DIR / "assets" / "qs_logo.png"


st.set_page_config(page_title="Chat de RR.HH.", page_icon="💬", layout="wide")

st.title("💬 Chat de RR.HH.")
st.caption("Pregunta lo que necesites sobre tu convenio laboral.")

# Sesión state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "top_k" not in st.session_state:
    st.session_state.top_k = 5

# Sidebar
with st.sidebar:
    st.image(
        LOGO_PATH,
        use_container_width=True,
    )
    st.header("Configuración")
    st.session_state.top_k = st.slider(
        "Top-K recuperación", 1, 10, st.session_state.top_k
    )
    st.markdown("---")
    st.write("**Backend API:**")
    st.code(API_URL)

    if st.button("Limpiar chat"):
        st.session_state.messages = []
        st.rerun()

# Render history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        # Show grounding only in assistant messages
        if msg["role"] == "assistant" and msg.get("grounding"):
            with st.expander("Ver grounding / fuentes"):
                for g in msg["grounding"]:
                    meta = (
                        f"**[{g['citation_id']}]** score={g['score']:.4f}  \n"
                        f"doc_id: `{g.get('doc_id')}`  \n"
                        f"source_file: `{g.get('source_file')}`  \n"
                        f"chunk_id: `{g.get('chunk_id')}`  \n"
                        f"pages: `{g.get('page_numbers', [])}`"
                    )
                    st.markdown(meta)
                    st.code(g.get("text", ""), language=None)
                    st.markdown("---")

# Input chat
prompt = st.chat_input("Escribe tu pregunta...")
if prompt:
    # User message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Backend call
    with st.chat_message("assistant"):
        with st.spinner("Buscando y generando respuesta..."):
            try:
                resp = requests.post(
                    f"{API_URL}/chat",
                    json={"query": prompt, "top_k": st.session_state.top_k},
                    timeout=120,
                )
                resp.raise_for_status()
                data = resp.json()

                answer = data.get("answer", "Sin respuesta.")
                grounding = data.get("grounding", [])

                st.markdown(answer)

                if grounding:
                    with st.expander("Ver grounding / fuentes", expanded=False):
                        for g in grounding:
                            meta = (
                                f"**[{g['citation_id']}]** score={g['score']:.4f}  \n"
                                f"doc_id: `{g.get('doc_id')}`  \n"
                                f"source_file: `{g.get('source_file')}`  \n"
                                f"chunk_id: `{g.get('chunk_id')}`  \n"
                                f"pages: `{g.get('page_numbers', [])}`"
                            )
                            st.markdown(meta)
                            st.code(g.get("text", ""), language=None)
                            st.markdown("---")

                # Save to history
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                        "grounding": grounding,
                    }
                )

            except requests.RequestException as e:
                err = f"Error llamando al backend: {e}"
                st.error(err)
                st.session_state.messages.append(
                    {"role": "assistant", "content": err, "grounding": []}
                )
