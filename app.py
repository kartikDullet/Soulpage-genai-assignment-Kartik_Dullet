import streamlit as st
from bot import ask_bot

st.set_page_config(page_title="Conversational Knowledge Bot", page_icon="🤖")
st.title("🤖 Conversational Knowledge Bot")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User input
if prompt := st.chat_input("Ask something..."):
    st.session_state.messages.append(
        {"role": "user", "content": prompt}
    )

    with st.chat_message("assistant"):
        response = ask_bot(prompt, session_id="streamlit")
        st.markdown(response["answer"])
        st.caption(f"Source: {response['source']}")

    st.session_state.messages.append(
        {"role": "assistant", "content": response["answer"]}
    )
