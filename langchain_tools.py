from langchain_core.tools import tool

from agent_tools import sql_tool, rag_tool


@tool
def sales_database_tool(question: str) -> dict:
    """Use this tool for sales, revenue, region, product, and customer questions."""

    return sql_tool(question)


@tool
def internal_document_tool(question: str) -> dict:
    """Use this tool for questions requiring information from internal documents."""

    return rag_tool(question)