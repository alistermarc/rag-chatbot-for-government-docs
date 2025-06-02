import streamlit as st
import requests

backend_url = "localhost"

# Initialize single chat session
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar: PDF Upload and Clear Chat
with st.sidebar:
    st.header("Chat Options")

    # Clear Chat button
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []

    st.divider()
    st.subheader("Upload PDFs (Batch Upload)")

    uploaded_files = st.file_uploader(
        "Choose one or more PDF files",
        type=["pdf"],
        accept_multiple_files=True
    )

    if uploaded_files and st.button("Upload and Process All"):
        files = [("file", (f.name, f, "application/pdf")) for f in uploaded_files]

        try:
            upload_response = requests.post(
                f"http://{backend_url}:8000/api/upload/",
                files=files
            )

            if upload_response.status_code == 207:
                result = upload_response.json()
                results = result.get("results", [])
                for res in results:
                    file = res.get("file", "Unknown file")
                    status = res.get("status", "unknown")
                    message = res.get("message", "")
                    
                    if status == "success":
                        st.success(f"{file}: {message}")
                    else:
                        st.error(f"{file}: {message}")
            else:
                try:
                    error_msg = upload_response.json().get("error", "Upload failed.")
                except:
                    error_msg = "Upload failed."
                st.error(error_msg)

        except Exception as e:
            st.error(f"Error during upload: {str(e)}")


# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and message.get("sources"):
            st.markdown(f"**Sources:** {', '.join(message['sources'])}")

# Handle user input
if user_input := st.chat_input("Ask about local ordinances and resolutions:"):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Clean chat history before sending to backend
    history = [
        {"role": msg["role"], "content": msg["content"]}
        for msg in st.session_state.messages
    ]

    # Backend API call
    response = requests.post(
        f'http://{backend_url}:8000/api/chat/',
        json={"message": user_input, "history": history}
    )

    if response.status_code == 200:
        data = response.json()
        final_response = data.get("answer", "No response from backend.")
        sources = data.get("sources", [])

        st.session_state.messages.append({
            "role": "assistant",
            "query": user_input,
            "content": final_response,
            "sources": sources
        })

        with st.chat_message("assistant"):
            st.write(final_response)
            if sources:
                st.markdown(f"**Sources:** {', '.join(sources)}")
    else:
        error_msg = "Error: Could not retrieve an answer."
        st.session_state.messages.append({
            "role": "assistant",
            "content": error_msg,
            "sources": []
        })
        with st.chat_message("assistant"):
            st.write(error_msg)
