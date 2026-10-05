from finassist_core import (
    route_question,
    run_sql_question,
    retrieve_rag_context,
    generate_rag_answer,
    generate_sql_answer,
    generate_combined_answer
)


def ask_question(question):

    route = route_question(question)

    print("\n" + "=" * 60)
    print("Question:", question)
    print("Route:", route)

    if route == "SQL":

        sql_results = run_sql_question(
            question
        )

        print("\nSQL result:")
        print(sql_results)

        answer = generate_sql_answer(
            question,
            sql_results
        )

    elif route == "RAG":

        context = retrieve_rag_context(
            question
        )

        print("\nRetrieved document context:")
        print(context)

        answer = generate_rag_answer(
            question,
            context
        )

    else:

        sql_results = run_sql_question(
            question
        )

        rag_context = retrieve_rag_context(
            question
        )

        print("\nSQL result:")
        print(sql_results)

        print("\nRetrieved document context:")
        print(rag_context)

        answer = generate_combined_answer(
            question,
            sql_results,
            rag_context
        )

    print("\nFinassist AI:")
    print(answer)


print("\nFinassist AI is ready.")
print("Type 'exit' to stop.")


while True:

    question = input("\nYou: ")

    if question.lower() == "exit":
        break

    if not question.strip():
        continue

    ask_question(question)