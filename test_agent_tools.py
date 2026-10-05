from agent_tools import (
    sql_tool,
    rag_tool
)


sql_result = sql_tool(
    "What is the total revenue in the South region?"
)

print("\nSQL TOOL RESULT:")
print(sql_result)


rag_result = rag_tool(
    "What is Generative AI?"
)

print("\nRAG TOOL RESULT:")
print(rag_result)