from __future__ import annotations

from typing import Any, Literal, TypedDict

from langgraph.graph import END, StateGraph

from ag_app.convenio_catalog import (
    CATALOGO_CONVENIOS,
    convenio_label,
    extract_provincia,
    find_by_sector_and_provincia,
    get_convenio_by_id,
    resolve_convenio_from_text,
)
from ag_app.rag_backend import answer_with_grounding


class AgentState(TypedDict, total=False):
    messages: list[dict[str, str]]
    last_user_message: str
    answer: str
    grounding: list[dict[str, Any]]
    top_k: int

    convenio_id: str | None
    convenio_label: str | None
    pending_sector: str | None

    awaiting_field: (
        Literal[
            "convenio",
            "provincia",
            "consulta",
            "confirmacion_sugerencia",
            "nueva_consulta",
        ]
        | None
    )

    options: list[str]
    candidate_ids: list[str]
    finished: bool


def _yes(text: str) -> bool:
    t = text.strip().lower()
    return t in {"si", "sí", "s", "vale", "ok", "claro", "por supuesto", "yes"}


def _no(text: str) -> bool:
    t = text.strip().lower()
    return t in {"no", "n", "nop", "ninguno"}


def _same_convenio(text: str) -> bool:
    t = text.strip().lower()
    return t in {
        "el mismo",
        "mismo",
        "el mismo convenio",
        "sobre el mismo",
        "mismo convenio",
        "igual",
        "ese mismo",
    }


def _new_convenio(text: str) -> bool:
    t = text.strip().lower()
    return t in {
        "otro",
        "otro convenio",
        "cambiar",
        "cambiar convenio",
        "uno nuevo",
        "distinto",
        "otro distinto",
    }


def welcome_node(state: AgentState) -> AgentState:
    if state.get("messages"):
        return state

    return {
        **state,
        "answer": (
            "¡Hola! Soy tu asistente de convenios laborales. "
            "¿Sobre qué convenio laboral quieres hacer una consulta?"
        ),
        "grounding": [],
        "awaiting_field": "convenio",
        "options": [],
        "candidate_ids": [],
        "finished": False,
    }


def parse_node(state: AgentState) -> AgentState:
    messages = state.get("messages", [])
    if not messages:
        return state

    last_user = ""
    for msg in reversed(messages):
        if msg["role"] == "user":
            last_user = msg["content"].strip()
            break

    return {
        **state,
        "last_user_message": last_user,
    }


def router_node(state: AgentState) -> AgentState:
    awaiting = state.get("awaiting_field")
    user_text = state.get("last_user_message", "")

    if awaiting == "provincia":
        provincia = extract_provincia(user_text)
        sector = state.get("pending_sector")

        if not provincia:
            return {
                **state,
                "answer": (
                    f"Necesito que me indiques la provincia del convenio de "
                    f"{sector}. Por ejemplo: Madrid, Málaga o Sevilla."
                ),
                "grounding": [],
                "awaiting_field": "provincia",
            }

        convenio = find_by_sector_and_provincia(sector, provincia)
        if not convenio:
            return {
                **state,
                "answer": (
                    f"No tengo ese convenio de {sector} para {provincia.title()}. "
                    "Estas son las opciones disponibles:"
                ),
                "grounding": [],
                "awaiting_field": "convenio",
                "options": [
                    convenio_label(c)
                    for c in CATALOGO_CONVENIOS
                    if c["sector"] == sector
                ],
                "candidate_ids": [
                    c["id"] for c in CATALOGO_CONVENIOS if c["sector"] == sector
                ],
            }

        return {
            **state,
            "convenio_id": convenio["id"],
            "convenio_label": convenio_label(convenio),
            "awaiting_field": "consulta",
            "answer": (
                f"Perfecto, trabajaremos con el convenio **{convenio_label(convenio)}**. "
                "¿Qué consulta quieres hacer?"
            ),
            "grounding": [],
            "options": [],
            "candidate_ids": [],
        }

    if awaiting == "confirmacion_sugerencia":
        if _yes(user_text):
            candidate_ids = state.get("candidate_ids", [])
            if not candidate_ids:
                return {
                    **state,
                    "answer": (
                        "No he podido recuperar la sugerencia anterior. "
                        "Indícame de nuevo el convenio que buscas."
                    ),
                    "grounding": [],
                    "awaiting_field": "convenio",
                    "options": [],
                }

            convenio = get_convenio_by_id(candidate_ids[0])
            if not convenio:
                return {
                    **state,
                    "answer": "No he podido recuperar el convenio sugerido.",
                    "grounding": [],
                    "awaiting_field": "convenio",
                }

            return {
                **state,
                "convenio_id": convenio["id"],
                "convenio_label": convenio_label(convenio),
                "awaiting_field": "consulta",
                "answer": (
                    f"Perfecto, trabajaremos con el convenio **{convenio_label(convenio)}**. "
                    "¿Qué consulta quieres hacer?"
                ),
                "grounding": [],
                "options": [],
                "candidate_ids": [],
            }

        if _no(user_text):
            return {
                **state,
                "answer": (
                    "Sentimos no haberte podido ayudar, estamos trabajando "
                    "para mejorar nuestra base de datos sobre convenios."
                ),
                "grounding": [],
                "awaiting_field": None,
                "finished": True,
                "options": [],
                "candidate_ids": [],
            }

        return {
            **state,
            "answer": 'Respóndeme por favor "sí" o "no".',
            "grounding": [],
            "awaiting_field": "confirmacion_sugerencia",
        }

    if awaiting == "nueva_consulta":
        if _yes(user_text):
            return {
                **state,
                "answer": (
                    "Perfecto. ¿Quieres hacer otra pregunta sobre el mismo convenio "
                    "o prefieres indicarme otro convenio?"
                ),
                "grounding": [],
                "awaiting_field": "same_or_new_convenio",
                "options": ["El mismo", "Otro convenio"],
                "candidate_ids": [],
            }

        if _no(user_text):
            return {
                **state,
                "answer": "Gracias. Cuando quieras, aquí estaré para ayudarte.",
                "grounding": [],
                "awaiting_field": None,
                "finished": True,
            }

        return {
            **state,
            "answer": 'Respóndeme por favor "sí" o "no". ¿Quieres hacer una nueva consulta?',
            "grounding": [],
            "awaiting_field": "nueva_consulta",
        }

    if awaiting == "same_or_new_convenio":
        if _same_convenio(user_text):
            convenio_id = state.get("convenio_id")
            convenio_label_value = state.get("convenio_label")

            if convenio_id and convenio_label_value:
                return {
                    **state,
                    "answer": (
                        f"Perfecto, seguimos con el convenio **{convenio_label_value}**. "
                        "¿Qué consulta quieres hacer?"
                    ),
                    "grounding": [],
                    "awaiting_field": "consulta",
                    "options": [],
                    "candidate_ids": [],
                }

            return {
                **state,
                "answer": (
                    "No tengo un convenio activo en este momento. "
                    "Indícame por favor qué convenio quieres consultar."
                ),
                "grounding": [],
                "awaiting_field": "convenio",
                "options": [],
                "candidate_ids": [],
            }

        if _new_convenio(user_text):
            return {
                **state,
                "answer": "Perfecto. Indícame qué convenio quieres consultar.",
                "grounding": [],
                "awaiting_field": "convenio",
                "convenio_id": None,
                "convenio_label": None,
                "pending_sector": None,
                "options": [],
                "candidate_ids": [],
            }

        return {
            **state,
            "answer": 'Respóndeme por favor "El mismo" o "Otro convenio".',
            "grounding": [],
            "awaiting_field": "same_or_new_convenio",
            "options": ["El mismo", "Otro convenio"],
        }

    if awaiting == "consulta":
        return state

    # Caso principal: resolver convenio a partir de texto libre
    result = resolve_convenio_from_text(user_text)
    status = result["status"]

    if status == "exact":
        convenio = result["convenio"]
        return {
            **state,
            "convenio_id": convenio["id"],
            "convenio_label": convenio_label(convenio),
            "awaiting_field": "consulta",
            "answer": (
                f"Tengo disponible el convenio **{convenio_label(convenio)}**. "
                "¿Qué consulta quieres hacer?"
            ),
            "grounding": [],
            "options": [],
            "candidate_ids": [],
        }

    if status == "needs_provincia":
        candidates = result.get("candidates", [])
        provincias = [c["provincia"].title() for c in candidates if c.get("provincia")]
        return {
            **state,
            "pending_sector": result["sector"],
            "awaiting_field": "provincia",
            "answer": (
                f"Tengo varias versiones del convenio de **{result['sector']}**. "
                f"¿De qué provincia lo necesitas? Opciones: {', '.join(provincias)}."
            ),
            "grounding": [],
            "options": provincias,
            "candidate_ids": [c["id"] for c in candidates],
        }

    if status == "candidates":
        candidates = result.get("candidates", [])
        if not candidates:
            return {
                **state,
                "answer": (
                    "Sentimos no haberte podido ayudar, estamos trabajando "
                    "para mejorar nuestra base de datos sobre convenios."
                ),
                "grounding": [],
                "finished": True,
                "awaiting_field": None,
            }

        labels = [convenio_label(c) for c in candidates]
        first = candidates[0]
        return {
            **state,
            "answer": (
                "No he encontrado exactamente ese convenio, pero tengo uno o varios parecidos:\n\n"
                + "\n".join(f"- {label}" for label in labels)
                + f"\n\n¿Quieres consultar sobre **{convenio_label(first)}**?"
            ),
            "grounding": [],
            "awaiting_field": "confirmacion_sugerencia",
            "options": labels,
            "candidate_ids": [c["id"] for c in candidates],
        }

    return {
        **state,
        "answer": (
            "Sentimos no haberte podido ayudar, estamos trabajando "
            "para mejorar nuestra base de datos sobre convenios."
        ),
        "grounding": [],
        "awaiting_field": None,
        "finished": True,
        "options": [],
        "candidate_ids": [],
    }


def answer_node(state: AgentState) -> AgentState:
    if state.get("awaiting_field") != "consulta":
        return state

    convenio_id = state.get("convenio_id")
    user_question = state.get("last_user_message", "")
    top_k = state.get("top_k", 5)

    # Si aún no hay convenio seleccionado, no respondemos
    if not convenio_id:
        return state

    convenio = get_convenio_by_id(convenio_id)
    if not convenio:
        return {
            **state,
            "answer": "No he podido localizar el convenio seleccionado.",
            "grounding": [],
            "awaiting_field": "convenio",
        }

    result = answer_with_grounding(
        query=user_question,
        top_k=top_k,
        source_files=convenio["source_files"],
    )

    answer = (
        f"Consulta sobre **{convenio_label(convenio)}**\n\n"
        f"{result['answer']}\n\n"
        "¿Quieres hacer una nueva consulta?"
    )

    return {
        **state,
        "answer": answer,
        "grounding": result["grounding"],
        "awaiting_field": "nueva_consulta",
        "options": ["Sí", "No"],
    }


def route_after_parse(state: AgentState) -> str:
    awaiting = state.get("awaiting_field")

    if awaiting == "consulta":
        return "answer"

    return "router"


def build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("welcome", welcome_node)
    graph.add_node("parse", parse_node)
    graph.add_node("router", router_node)
    graph.add_node("answer", answer_node)

    graph.set_entry_point("welcome")
    graph.add_edge("welcome", "parse")
    graph.add_conditional_edges(
        "parse",
        route_after_parse,
        {
            "router": "router",
            "answer": "answer",
        },
    )
    graph.add_edge("router", END)
    graph.add_edge("answer", END)

    return graph.compile()


agent_app = build_graph()


def run_agent_turn(
    messages: list[dict[str, str]],
    top_k: int = 5,
    current_state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    initial_state: AgentState = {
        "messages": messages,
        "top_k": top_k,
        **(current_state or {}),
    }
    result = agent_app.invoke(initial_state)

    return {
        "answer": result.get("answer", ""),
        "grounding": result.get("grounding", []),
        "state": {
            "convenio_id": result.get("convenio_id"),
            "convenio_label": result.get("convenio_label"),
            "pending_sector": result.get("pending_sector"),
            "awaiting_field": result.get("awaiting_field"),
            "options": result.get("options", []),
            "candidate_ids": result.get("candidate_ids", []),
            "finished": result.get("finished", False),
        },
    }
