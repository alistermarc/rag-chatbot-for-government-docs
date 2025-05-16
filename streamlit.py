import streamlit as st
import requests

# Initialize session state
if "chats" not in st.session_state:
    st.session_state.chats = {
        "default": {
            "messages": [],
            "sources": set()
        }
    }
if "current_chat" not in st.session_state:
    st.session_state.current_chat = "default"

# Sidebar for chat management
with st.sidebar:
    st.header("Chat History")
    # New Chat button
    if st.button("➕ New Chat"):
        chat_id = f"chat_{len(st.session_state.chats) + 1}"
        st.session_state.chats[chat_id] = {
            "messages": [],
            "sources": set()
        }
        st.session_state.current_chat = chat_id

    # Display chat history
    for chat_id in st.session_state.chats:
        if st.button(chat_id.capitalize().replace("_", " ")):
            st.session_state.current_chat = chat_id

    st.divider()
    st.subheader("Upload PDF")

    uploaded_file = st.file_uploader("Choose a PDF", type=["pdf"])
    if uploaded_file is not None:
        if st.button("Upload and Process"):
            # files = {"file": uploaded_file.getvalue()}
            upload_response = requests.post(
                "http://127.0.0.1:8000/api/upload/",
                files={"file": (uploaded_file.name, uploaded_file, "application/pdf")}
            )
            if upload_response.status_code == 201:
                result = upload_response.json()
                st.success(result["message"])
                # Add to current chat's sources
                st.session_state.chats[st.session_state.current_chat]["sources"].add(result["source"])
            else:
                try:
                    error_msg = upload_response.json().get("error", "Upload failed.")
                except:
                    error_msg = "Upload failed."
                st.error(error_msg)

# Main chat interface
current_chat = st.session_state.chats[st.session_state.current_chat]

# Display messages for current chat
for message in current_chat["messages"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and message.get("sources"):
            formatted_sources = [f"{src}" for src in message['sources']]
            st.markdown(f"**Sources:** {', '.join(formatted_sources)}")

# Handle new input
if user_input := st.chat_input("Ask about local ordinances and resolutions:"):
    # Add user message to current chat
    current_chat["messages"].append({"role": "user", "content": user_input})
    
    with st.chat_message("user"):
        st.markdown(user_input)

    # Call Django backend API
    response = requests.post(
        'http://127.0.0.1:8000/api/chat/',
        json={"message": user_input, "history": current_chat["messages"]}
    )

    if response.status_code == 200:
        data = response.json()
        final_response = data.get("answer", "No response from backend.")
        sources = data.get("sources", [])

        # Add assistant response to current chat
        current_chat["messages"].append({
            "role": "assistant",
            "query": user_input,
            "content": final_response,
            "sources": sources
        })

        # Display assistant response
        with st.chat_message("assistant"):
            st.write(final_response)
            if sources:
                st.markdown(f"\n**Verified Sources:** {', '.join(sources)}")
    else:
        error_msg = "Error: Could not retrieve an answer."
        current_chat["messages"].append({
            "role": "assistant",
            "content": error_msg,
            "sources": []
        })
        with st.chat_message("assistant"):
            st.write(error_msg)
