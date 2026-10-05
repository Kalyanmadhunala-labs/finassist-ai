import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent

from langchain_tools import (
    sales_database_tool,
    internal_document_tool
)

load_dotenv()

llm = ChatOpenAI(
    model="openai/gpt-oss-20b",
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)

tools = [
    sales_database_tool,
    internal_document_tool
]

agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt="""
You are Finassist AI, an enterprise business intelligence assistant.

Use the sales database tool for questions about revenue,
sales, regions, products, customers, and transactions.

Use the internal document tool for questions about information
contained in internal documents.

If a question requires both business data and document information,
use both tools.

Use tool results as the source of truth.
Do not invent business data.

Revenue values from the sales database are in Indian Rupees.
Always use the ₹ symbol for revenue.
"""
)

print("\nFinassist AI LangChain Agent")
print("Type 'exit' to stop.")

while True:
    question = input("\nYou: ")

    if question.strip().lower() in ["exit", "quit"]:
        break

    if not question.strip():
        continue

    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": question
                }
            ]
        }
    )

    print("\nFinassist AI:")
    print(result["messages"][-1].content)