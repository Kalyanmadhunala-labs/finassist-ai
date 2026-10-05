from langchain_tools import sales_database_tool


result = sales_database_tool.invoke(
    {
        "question": "What is the total revenue in the South region?"
    }
)

print(result)