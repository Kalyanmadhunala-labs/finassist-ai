import os

from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()


client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)


def route_question(question):

    prompt = f"""
You are the routing component of an enterprise AI system.

Classify the user's question into exactly one category:

SQL
RAG
BOTH

Choose SQL when the question requires structured business
data, calculations, aggregations, filtering, grouping, or
information from the sales database.

Choose RAG when the question requires information from
documents or knowledge stored in the document database.

Choose BOTH when the question requires information from
both structured business data and documents.

Return only one word:
SQL
RAG
or
BOTH

User question:
{question}
"""

    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    result = (
        response
        .choices[0]
        .message
        .content
        .strip()
        .upper()
    )

    if result not in ["SQL", "RAG", "BOTH"]:
        return "RAG"

    return result