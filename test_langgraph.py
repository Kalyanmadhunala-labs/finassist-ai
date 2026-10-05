from typing import TypedDict

from langgraph.graph import StateGraph, START, END


class State(TypedDict):
    message: str


def first_node(state: State):
    return {
        "message": state["message"] + " → First node completed"
    }


def second_node(state: State):
    return {
        "message": state["message"] + " → Second node completed"
    }


graph = StateGraph(State)

graph.add_node("first", first_node)
graph.add_node("second", second_node)

graph.add_edge(START, "first")
graph.add_edge("first", "second")
graph.add_edge("second", END)

app = graph.compile()

result = app.invoke(
    {
        "message": "Start"
    }
)

print(result)