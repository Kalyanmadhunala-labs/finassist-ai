import os

import psycopg
from dotenv import load_dotenv
from openai import OpenAI
from pgvector.psycopg import register_vector
from sentence_transformers import SentenceTransformer, CrossEncoder


load_dotenv()


conn = psycopg.connect(
    host=os.getenv("DB_HOST"),
    port=os.getenv("DB_PORT"),
    dbname=os.getenv("DB_NAME"),
    user=os.getenv("DB_USER"),
    password=os.getenv("DB_PASSWORD")
)

register_vector(conn)


embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

reranker = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)


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
documents.

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


def run_sql_question(question):

    prompt = f"""
Generate one PostgreSQL SELECT query for the structured
business-data part of the user's question.

Ignore any request for information from documents,
knowledge bases, RAG, or other unstructured sources.

Database table:

sales_transactions(
    transaction_id,
    transaction_date,
    region,
    product,
    customer_segment,
    quantity,
    revenue
)

Return only the SQL query.

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

    query = (
        response
        .choices[0]
        .message
        .content
        .strip()
    )

    query = query.replace("```sql", "")
    query = query.replace("```", "")
    query = query.strip()

    query_lower = query.lower()

    blocked_keywords = [
        "insert",
        "update",
        "delete",
        "drop",
        "alter",
        "truncate",
        "create",
        "grant",
        "revoke"
    ]

    if not query_lower.startswith("select"):
        return {
            "status": "ERROR",
            "message": "Only SELECT queries are allowed."
        }

    if ";" in query_lower[:-1]:
        return {
            "status": "ERROR",
            "message": "Multiple SQL statements are not allowed."
        }

    for keyword in blocked_keywords:

        if keyword in query_lower:
            return {
                "status": "ERROR",
                "message": f"Blocked SQL operation: {keyword}"
            }

    with conn.cursor() as cur:

        cur.execute(query)

        results = cur.fetchall()

    if not results:
        return {
            "status": "NO_DATA",
            "message": "No matching records were found."
        }

    return {
        "status": "DATA_FOUND",
        "result": results
    }


def retrieve_rag_context(question):

    question_embedding = embedding_model.encode(
        question
    ).astype("float32")

    with conn.cursor() as cur:

        cur.execute(
            """
            SELECT
                chunk_id,
                content
            FROM document_chunks
            ORDER BY embedding <=> %s
            LIMIT 5
            """,
            (question_embedding,)
        )

        vector_results = cur.fetchall()

    if not vector_results:
        return ""

    pairs = []

    for chunk_id, content in vector_results:

        pairs.append(
            [question, content]
        )

    scores = reranker.predict(pairs)

    ranked_results = []

    for i, result in enumerate(vector_results):

        chunk_id, content = result

        ranked_results.append(
            (
                chunk_id,
                content,
                float(scores[i])
            )
        )

    ranked_results.sort(
        key=lambda x: x[2],
        reverse=True
    )

    top_results = ranked_results[:2]

    context_parts = []

    for chunk_id, content, score in top_results:

        context_parts.append(content)

    return "\n\n".join(context_parts)


def generate_rag_answer(question, context):

    prompt = f"""
Answer the user's question using only the context.

If the answer is not available in the context,
say that the information is not available.

Context:
{context}

Question:
{question}

Answer:
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

    return (
        response
        .choices[0]
        .message
        .content
        .strip()
    )


def generate_sql_answer(question, sql_results):

    prompt = f"""
Answer the user's business question using the SQL result.

The revenue values are in Indian Rupees (INR).
Use the ₹ symbol when presenting revenue.

Important:

- If the SQL result status is NO_DATA, clearly state
  that no matching records were found.
- Do not convert NO_DATA into ₹0.
- Do not invent a value.
- If the SQL result status is ERROR, explain that the
  database query could not be executed.

Question:
{question}

SQL result:
{sql_results}

Give a clear, concise business answer.
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

    return (
        response
        .choices[0]
        .message
        .content
        .strip()
    )


def generate_combined_answer(
    question,
    sql_results,
    rag_context
):

    prompt = f"""
You are the final synthesis component of an enterprise
AI decision intelligence system.

Answer the user's question using BOTH sources below.

Structured business data from PostgreSQL:
{sql_results}

Information retrieved from internal documents:
{rag_context}

Rules:

- Use the PostgreSQL result for numerical business facts.
- Use the document context for document-based information.
- Do not invent information.
- If the SQL status is NO_DATA, say that no matching
  records were found. Do not report ₹0 unless the data
  explicitly shows zero revenue.
- If one source does not contain the required information,
  clearly say that it is not available.
- Revenue values are in Indian Rupees (INR).
- Use the ₹ symbol for revenue.
- Combine the information into one clear business answer.

User question:
{question}

Final answer:
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

    return (
        response
        .choices[0]
        .message
        .content
        .strip()
    )