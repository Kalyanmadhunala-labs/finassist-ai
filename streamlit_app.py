import requests
import streamlit as st

st.set_page_config(
    page_title="Finassist AI",
    page_icon="🤖",
    layout="centered"
)

st.title("Finassist AI")
st.subheader("Enterprise AI Decision Intelligence")

question = st.text_area(
    "Ask a business question",
    placeholder="Example: What is the total revenue in the South region?"
)

if st.button("Ask Finassist AI"):

    if not question.strip():
        st.warning("Please enter a question.")
    else:
        try:
            response = requests.post(
                "http://127.0.0.1:8000/ask",
                json={"question": question},
                timeout=120
            )

            if response.status_code == 200:
                result = response.json()

                st.success("Answer generated")

                st.subheader("Answer")
                st.write(result["answer"])

                st.caption(
                    f"Response time: {result['response_time_seconds']} seconds"
                )

            else:
                st.error(
                    f"API request failed with status code {response.status_code}"
                )

        except requests.exceptions.RequestException:
            st.error(
                "Could not connect to Finassist AI API. "
                "Make sure the FastAPI server is running."
            )