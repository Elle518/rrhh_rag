"""Streamlit app for the RAG-based convenio chatbot. This app provides a user interface to interact with the backend API, send user queries, and display the assistant's responses along with the retrieved grounding information. The app also manages the chat history and agent state across interactions, allowing for a conversational experience.

To run the app, use the following commands in separate terminals:

1. > uvicorn ag_app.api:app --reload --host 0.0.0.0 --port 8000

2. > streamlit run ag_app/streamlit_app.py
"""

from __future__ import annotations

import logging
import uuid
from pathlib import Path

import requests
import streamlit as st
from dotenv import load_dotenv

from rrhh_rag import conf

logger = logging.getLogger(__name__)

load_dotenv()

try:
    conf_settings = conf.load(file="settings.yaml")
except Exception as e:
    logger.error("Failed to load conf files: %s", e)
    raise

API_URL = conf_settings.rag_api_url

BASE_DIR = Path(__file__).resolve().parent
LOGO_PATH = BASE_DIR / "assets" / "qs_logo.png"

st.set_page_config(page_title="Chat de Convenios", page_icon="💬", layout="wide")
st.title("💬 Chat de Convenios")
st.caption(
    "Consulta información sobre convenios laborales disponibles en la base de datos."
)

if "messages" not in st.session_state:
    st.session_state.messages = []

if "top_k" not in st.session_state:
    st.session_state.top_k = 5

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "agent_state" not in st.session_state:
    st.session_state.agent_state = {
        "convenio_id": None,
        "convenio_label": None,
        "pending_sector": None,
        "awaiting_field": None,
        "options": [],
        "candidate_ids": [],
        "finished": False,
    }


def render_grounding(grounding: list[dict]):
    """Render the grounding information retrieved from the backend in a user-friendly format. Each grounding item includes metadata such as citation ID, score, source file, and the text of the retrieved chunk. The grounding is displayed in an expandable section to keep the chat interface clean."""
    if not grounding:
        return
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


def send_message(user_text: str):
    """Send a user message to the backend API and handle the response, updating the chat history and agent state accordingly."""
    st.session_state.messages.append({"role": "user", "content": user_text})

    try:
        resp = requests.post(
            f"{API_URL}/chat",
            json={
                "session_id": st.session_state.session_id,
                "message": user_text,
                "top_k": st.session_state.top_k,
            },
            timeout=120,
        )
        resp.raise_for_status()
        data = resp.json()

        answer = data.get("answer", "Sin respuesta.")
        grounding = data.get("grounding", [])
        state = data.get("state", {})

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer,
                "grounding": grounding,
            }
        )
        st.session_state.agent_state = state

    except requests.RequestException as e:
        err = f"Error llamando al backend: {e}"
        st.session_state.messages.append(
            {"role": "assistant", "content": err, "grounding": []}
        )


with st.sidebar:
    st.image(LOGO_PATH, width="content")
    st.header("Configuración")
    st.session_state.top_k = st.slider(
        "Top-K recuperación", 1, 10, st.session_state.top_k
    )

    if st.session_state.agent_state.get("convenio_label"):
        st.success(f"Convenio activo: {st.session_state.agent_state['convenio_label']}")

    if st.button("Limpiar chat"):
        try:
            requests.post(f"{API_URL}/reset/{st.session_state.session_id}", timeout=30)
        except Exception:
            pass

        st.session_state.messages = []
        st.session_state.agent_state = {
            "convenio_id": None,
            "convenio_label": None,
            "pending_sector": None,
            "awaiting_field": None,
            "options": [],
            "candidate_ids": [],
            "finished": False,
        }
        st.session_state.session_id = str(uuid.uuid4())
        st.rerun()


# Welcome message
if not st.session_state.messages:
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": (
                "¡Hola! Soy tu asistente de convenios laborales. "
                "¿Sobre qué convenio laboral quieres hacer una consulta?"
            ),
            "grounding": [],
        }
    )
    st.session_state.agent_state = {
        "convenio_id": None,
        "convenio_label": None,
        "pending_sector": None,
        "awaiting_field": "convenio",
        "options": [],
        "candidate_ids": [],
        "finished": False,
    }

# render history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            render_grounding(msg.get("grounding", []))

# Agent options
options = st.session_state.agent_state.get("options") or []
awaiting_field = st.session_state.agent_state.get("awaiting_field")

if options and awaiting_field in {
    "provincia",
    "confirmacion_sugerencia",
    "nueva_consulta",
    "convenio",
}:
    st.markdown("### Opciones sugeridas")
    cols = st.columns(min(len(options), 3))
    for idx, option in enumerate(options):
        with cols[idx % len(cols)]:
            if st.button(option, key=f"option_{idx}_{option}"):
                send_message(option)
                st.rerun()

prompt = st.chat_input("Escribe tu mensaje...")
if prompt:
    with st.chat_message("user"):
        st.markdown(prompt)
    send_message(prompt)
    st.rerun()
