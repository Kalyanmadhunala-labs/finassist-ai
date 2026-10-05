from finassist_core import (
    run_sql_question,
    retrieve_rag_context
)


def sql_tool(question):

    result = run_sql_question(
        question
    )

    return {
        "tool": "sql_tool",
        "result": result
    }


def rag_tool(question):

    result = retrieve_rag_context(
        question
    )

    return {
        "tool": "rag_tool",
        "result": result
    }