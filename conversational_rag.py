import os
import json

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

print("Connected to PostgreSQL")


embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

print("Embedding model loaded")


reranker = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)

print("Reranker model loaded")


client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)

print("Groq client connected")


chat_history = []


def rewrite_question(question, chat_history):

    if not chat_history:
        return question

    history_text = ""

    for message in chat_history:
        history_text += (
            f"{message['role']}: "
            f"{message['content']}\n"
        )

    rewrite_prompt = f"""
You are a question rewriting component.

Rewrite the latest user question into a standalone
search query using the conversation history.

Do not answer the question.

Return only the rewritten question.

Conversation history:
{history_text}

Latest user question:
{question}

Standalone question:
"""

    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {
                "role": "user",
                "content": rewrite_prompt
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


def extract_metadata(question):

    metadata_prompt = f"""
Extract metadata filters explicitly mentioned or clearly
specified in the user's question.

Return ONLY valid JSON with exactly these fields:

{{
    "year": null,
    "region": null,
    "document_type": null
}}

Rules:

- year must be an integer if a year is mentioned.
- region must be the region mentioned by the user.
- document_type must be the document type mentioned by the user.
- If a value is not mentioned, return null.
- Do not guess values.
- Do not add extra fields.

User question:
{question}
"""

    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {
                "role": "user",
                "content": metadata_prompt
            }
        ]
    )

    metadata_text = (
        response
        .choices[0]
        .message
        .content
        .strip()
    )

    try:
        metadata = json.loads(metadata_text)
    except json.JSONDecodeError:
        metadata = {
            "year": None,
            "region": None,
            "document_type": None
        }

    return metadata


def retrieve_context(question, metadata):

    question_embedding = embedding_model.encode(
        question
    ).astype("float32")

    conditions = []
    parameters = []

    if metadata.get("year") is not None:
        conditions.append("year = %s")
        parameters.append(metadata["year"])

    if metadata.get("region") is not None:
        conditions.append("region = %s")
        parameters.append(metadata["region"])

    if metadata.get("document_type") is not None:
        conditions.append("document_type = %s")
        parameters.append(metadata["document_type"])

    filter_clause = ""

    if conditions:
        filter_clause = "WHERE " + " AND ".join(conditions)

    vector_query = f"""
        SELECT
            chunk_id,
            content
        FROM document_chunks
        {filter_clause}
        ORDER BY embedding <=> %s
        LIMIT 5
    """

    vector_parameters = (
        parameters + [question_embedding]
    )

    with conn.cursor() as cur:

        cur.execute(
            vector_query,
            vector_parameters
        )

        vector_results = cur.fetchall()

    keyword_conditions = list(conditions)
    keyword_parameters = list(parameters)

    keyword_conditions.append(
        """
        search_vector @@
        plainto_tsquery('english', %s)
        """
    )

    keyword_parameters.append(question)

    keyword_filter_clause = (
        "WHERE " +
        " AND ".join(keyword_conditions)
    )

    keyword_query = f"""
        SELECT
            chunk_id,
            content
        FROM document_chunks
        {keyword_filter_clause}
        LIMIT 5
    """

    with conn.cursor() as cur:

        cur.execute(
            keyword_query,
            keyword_parameters
        )

        keyword_results = cur.fetchall()

    candidates = {}

    for chunk_id, content in vector_results:
        candidates[chunk_id] = content

    for chunk_id, content in keyword_results:
        candidates[chunk_id] = content

    if not candidates:
        return []

    pairs = []

    for content in candidates.values():
        pairs.append(
            [question, content]
        )

    reranker_scores = reranker.predict(
        pairs
    )

    ranked_results = []

    candidate_items = list(
        candidates.items()
    )

    for i, (chunk_id, content) in enumerate(
        candidate_items
    ):

        score = float(
            reranker_scores[i]
        )

        ranked_results.append(
            (
                chunk_id,
                content,
                score
            )
        )

    ranked_results.sort(
        key=lambda x: x[2],
        reverse=True
    )

    return ranked_results[:2]


def generate_answer(question, context):

    prompt = f"""
Answer the user's question using only the context
provided below.

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


def ask_question(question):

    print("\n" + "=" * 60)

    print("User question:")
    print(question)

    standalone_question = rewrite_question(
        question,
        chat_history
    )

    print("\nStandalone question:")
    print(standalone_question)

    metadata = extract_metadata(
        standalone_question
    )

    print("\nExtracted metadata:")
    print(
        json.dumps(
            metadata,
            indent=2
        )
    )

    top_results = retrieve_context(
        standalone_question,
        metadata
    )

    print("\nRetrieved context:")

    if not top_results:

        print("No matching documents found.")

        answer = (
            "The information is not available."
        )

    else:

        for chunk_id, content, score in top_results:

            print(
                f"\nChunk ID: {chunk_id}"
            )

            print(
                f"Reranker score: {score:.4f}"
            )

            print(
                f"Content: {content}"
            )

        context_parts = []

        for chunk_id, content, score in top_results:

            context_parts.append(
                content
            )

        context = "\n\n".join(
            context_parts
        )

        answer = generate_answer(
            question,
            context
        )

    print("\nAssistant:")
    print(answer)

    chat_history.append(
        {
            "role": "user",
            "content": question
        }
    )

    chat_history.append(
        {
            "role": "assistant",
            "content": answer
        }
    )


print("\n" + "=" * 60)
print("Finassist AI Conversational RAG")
print("Type 'exit' to stop the chatbot.")
print("=" * 60)


while True:

    user_question = input("\nYou: ")

    if user_question.lower() == "exit":
        break

    if not user_question.strip():
        continue

    ask_question(user_question)


conn.close()

print("\nDatabase connection closed.")