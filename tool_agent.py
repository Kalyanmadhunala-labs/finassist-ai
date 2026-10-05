import json
import os

from dotenv import load_dotenv
from openai import OpenAI

from agent_tools import sql_tool, rag_tool


load_dotenv()


client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)


tools = [
    {
        "type": "function",
        "function": {
            "name": "sql_tool",
            "description": "Use this tool for structured business data, revenue calculations, filtering, grouping, and sales database questions.",
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "The business question that should be answered using the sales database."
                    }
                },
                "required": ["question"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "rag_tool",
            "description": "Use this tool for questions that require information from internal documents or knowledge stored in the document database.",
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "The question that should be answered using internal documents."
                    }
                },
                "required": ["question"]
            }
        }
    }
]


def run_tool(name, arguments):

    if name == "sql_tool":

        result = sql_tool(
            arguments["question"]
        )

        from validation import validate_sql_result

        if not validate_sql_result(result):
            return {
                "tool": "sql_tool",
                "status": "invalid",
                "result": result
            }

        return {
            "tool": "sql_tool",
            "status": "valid",
            "result": result
        }

    if name == "rag_tool":

        result = rag_tool(
            arguments["question"]
        )

        from validation import validate_rag_result

        if not validate_rag_result(result):
            return {
                "tool": "rag_tool",
                "status": "invalid",
                "result": result
            }

        return {
            "tool": "rag_tool",
            "status": "valid",
            "result": result
        }

    return {
        "tool": "unknown",
        "status": "invalid",
        "result": "Unknown tool"
    }


def ask_agent(question):

    messages = [
        {
            "role": "system",
            "content": """
You are Finassist AI, an enterprise business intelligence assistant.

Follow these rules:

- Revenue and monetary values from the sales database are in Indian Rupees.
- Always use the ₹ symbol for sales revenue.
- Never use $, USD, or dollar symbols for sales revenue.
- Use tool results as the source of truth.
- Do not invent business data.
- Use rag_tool for internal document questions.
- Use sql_tool for structured business-data questions.
- If the question requires both, use both tools.
- Do not call the same tool more than once.
"""
        },
        {
            "role": "user",
            "content": question
        }
    ]

    used_tools = set()
    collected_results = {}

    for _ in range(3):

        available_tools = [
            tool
            for tool in tools
            if tool["function"]["name"] not in used_tools
        ]

        if not available_tools:
            break

        response = client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=messages,
            tools=available_tools,
            tool_choice="auto"
        )

        message = response.choices[0].message

        if not message.tool_calls:
            return message.content

        messages.append(message)

        for tool_call in message.tool_calls:

            tool_name = tool_call.function.name

            arguments = json.loads(
                tool_call.function.arguments
            )

            print("\nTool selected:", tool_name)
            print("Tool arguments:", arguments)

            if tool_name in used_tools:

                tool_result = {
                    "tool": tool_name,
                    "status": "already_executed",
                    "result": "This tool was already executed. Use the existing result."
                }

            else:

                used_tools.add(tool_name)

                tool_result = run_tool(
                    tool_name,
                    arguments
                )

                collected_results[tool_name] = tool_result

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(
                        tool_result,
                        default=str
                    )
                }
            )

    synthesis_prompt = f"""
You are the final answer generator for Finassist AI.

Answer the user's original question using ALL relevant tool results below.

Original user question:
{question}

Tool results:
{json.dumps(collected_results, default=str, indent=2)}

Rules:

- Use the SQL result for numerical business facts.
- Use the RAG result for information from internal documents.
- If both tools were used, include information from BOTH tools.
- Do not ignore a relevant tool result.
- Do not invent information that is not present in the tool results.
- Revenue values are in Indian Rupees.
- Always use the ₹ symbol for revenue.
- If a tool returned no matching data, clearly say that no matching records were found.
- Give the answer in a clear and concise business format.
"""

    final_response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {
                "role": "system",
                "content": "You are the final synthesis component of Finassist AI."
            },
            {
                "role": "user",
                "content": synthesis_prompt
            }
        ]
    )

    return (
        final_response
        .choices[0]
        .message
        .content
    )


print("\nFinassist AI Tool Agent")
print("Type 'exit' to stop.")


while True:

    question = input("\nYou: ")

    if question.strip().lower() in ["exit", "quit"]:
      break

    if not question.strip():
        continue

    answer = ask_agent(question)

    print("\nFinassist AI:")
    print(answer)