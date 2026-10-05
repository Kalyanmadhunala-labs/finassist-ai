from typing import TypedDict
import json
import os
import logging
import time
import uuid

from dotenv import load_dotenv
from openai import OpenAI
from langgraph.graph import StateGraph, START, END

from router import route_question
from agent_tools import sql_tool, rag_tool
from validation import validate_sql_result, validate_rag_result


load_dotenv()


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)


class FinassistState(TypedDict):
    question: str
    route: str
    sql_result: dict
    rag_result: str
    validation_result: bool
    final_answer: str


def router_node(state: FinassistState):
    start_time = time.time()

    route = route_question(state["question"])

    elapsed = time.time() - start_time

    logger.info(
        "Route selected | route=%s | duration=%.3fs",
        route,
        elapsed
    )

    return {
        "route": route
    }


def sql_node(state: FinassistState):
    start_time = time.time()

    result = sql_tool(state["question"])

    elapsed = time.time() - start_time

    logger.info(
        "SQL node completed | duration=%.3fs",
        elapsed
    )

    return {
        "sql_result": result
    }


def rag_node(state: FinassistState):
    start_time = time.time()

    result = rag_tool(state["question"])

    elapsed = time.time() - start_time

    logger.info(
        "RAG node completed | duration=%.3fs",
        elapsed
    )

    return {
        "rag_result": result
    }


def route_after_router(state: FinassistState):
    return state["route"]


def route_after_sql(state: FinassistState):
    if state["route"] == "BOTH":
        return "rag"

    return "validation"


def validation_node(state: FinassistState):
    start_time = time.time()

    if state["route"] == "SQL":
        valid = validate_sql_result(state["sql_result"])

    elif state["route"] == "RAG":
        valid = validate_rag_result(state["rag_result"])

    elif state["route"] == "BOTH":
        sql_valid = validate_sql_result(state["sql_result"])
        rag_valid = validate_rag_result(state["rag_result"])

        valid = sql_valid and rag_valid

    else:
        valid = False

    elapsed = time.time() - start_time

    logger.info(
        "Validation completed | result=%s | duration=%.3fs",
        valid,
        elapsed
    )

    return {
        "validation_result": valid
    }


def route_after_validation(state: FinassistState):
    if state["validation_result"]:
        return "synthesis"

    return "error"


def error_node(state: FinassistState):
    logger.error("Error handling node executed")

    return {
        "final_answer": (
            "I could not validate the retrieved data, "
            "so I cannot provide a reliable answer."
        )
    }


def synthesis_node(state: FinassistState):
    start_time = time.time()

    prompt = f"""
You are the final answer generator for Finassist AI.

Answer the user's original question using the available
tool results below.

Original question:
{state["question"]}

SQL result:
{json.dumps(state["sql_result"], default=str, indent=2)}

RAG result:
{json.dumps(state["rag_result"], default=str, indent=2)}

Rules:

- Use SQL results as the source of truth for business numbers.
- Use RAG results as the source of truth for internal document information.
- Do not invent information.
- Revenue values are in Indian Rupees.
- Always use the ₹ symbol for revenue.
- If both SQL and RAG results are available, use both.
- Give a clear and concise business answer.
"""

    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are the final synthesis component "
                    "of Finassist AI."
                )
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    final_answer = response.choices[0].message.content

    elapsed = time.time() - start_time

    logger.info(
        "Synthesis completed | duration=%.3fs",
        elapsed
    )

    return {
        "final_answer": final_answer
    }


def run_finassist(state: FinassistState):
    request_id = str(uuid.uuid4())[:8]
    start_time = time.time()

    logger.info(
        "Request started | request_id=%s | question=%s",
        request_id,
        state["question"]
    )

    try:
        result = app.invoke(state)

        elapsed = time.time() - start_time

        logger.info(
            "Request completed | request_id=%s | duration=%.3fs",
            request_id,
            elapsed
        )

        return result

    except Exception:
        elapsed = time.time() - start_time

        logger.exception(
            "Request failed | request_id=%s | duration=%.3fs",
            request_id,
            elapsed
        )

        raise


graph = StateGraph(FinassistState)


graph.add_node("router", router_node)
graph.add_node("sql", sql_node)
graph.add_node("rag", rag_node)
graph.add_node("validation", validation_node)
graph.add_node("synthesis", synthesis_node)
graph.add_node("error", error_node)


graph.add_edge(START, "router")


graph.add_conditional_edges(
    "router",
    route_after_router,
    {
        "SQL": "sql",
        "RAG": "rag",
        "BOTH": "sql",
    }
)


graph.add_conditional_edges(
    "sql",
    route_after_sql,
    {
        "rag": "rag",
        "validation": "validation",
    }
)


graph.add_edge("rag", "validation")


graph.add_conditional_edges(
    "validation",
    route_after_validation,
    {
        "synthesis": "synthesis",
        "error": "error",
    }
)


graph.add_edge("synthesis", END)
graph.add_edge("error", END)


app = graph.compile()