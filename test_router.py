from router import route_question


questions = [
    "What is the total revenue in the South region?",
    "What is Generative AI?",
    "What is RAG?",
    "What was the revenue in South and what does our document say about RAG?"
]


for question in questions:

    route = route_question(question)

    print("\nQuestion:", question)
    print("Route:", route)