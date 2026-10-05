# Finassist AI

## Enterprise AI Decision Intelligence Platform

Finassist AI is an enterprise-style AI decision intelligence platform that combines **structured business data, internal documents, LLMs, RAG, vector search, and workflow orchestration** to answer business questions with grounded and validated responses.

The system intelligently determines whether a question requires:

- **SQL** — structured business data
- **RAG** — internal document knowledge
- **BOTH** — structured data + internal documents

The selected workflow is executed through LangGraph, validated, and synthesized into a final business-friendly answer.

---

# 1. Project Overview

Traditional enterprise AI systems often need to work with multiple sources of information.

For example:

- Business metrics may exist in PostgreSQL.
- Policies, documentation, and knowledge may exist in internal documents.
- Users expect to ask questions using natural language rather than SQL.
- LLM-generated answers need validation and grounding.

Finassist AI addresses this problem by providing a unified AI workflow that can reason over both structured and unstructured enterprise information.

### Example

A user can ask:

> What is the total revenue in the South region?

The system identifies that this is a structured-data question and routes it to the SQL workflow.

A user can also ask:

> What does our internal document say about RAG?

The system identifies that this requires document knowledge and routes it to the RAG workflow.

The system can also handle combined questions:

> What is the total revenue in the South region, and what does our internal document say about RAG?

This requires both SQL and RAG, so the system executes both workflows and combines their results.

---

# 2. High-Level Architecture

```text
                         User
                           |
                           v
                    FastAPI /ask
                           |
                           v
                    LangGraph Workflow
                           |
                           v
                      AI Router
                     /    |     \
                   SQL   RAG    BOTH
                    |     |       |
                    v     v       v
              PostgreSQL  pgvector
                    |       |
                    |       v
                    |   Vector Retrieval
                    |       |
                    |       v
                    |   Cross-Encoder
                    |     Reranking
                    |       |
                    +---+---+
                        |
                        v
                    Validation
                        |
                        v
                    Synthesis
                        |
                        v
                   Final Answer