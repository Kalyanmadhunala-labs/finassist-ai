from langchain_tools import internal_document_tool

result = internal_document_tool.invoke(
    {
        "question": "What is Generative AI?"
    }
)

print(result)