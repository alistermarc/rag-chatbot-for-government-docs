import streamlit as st
from transformers import AutoTokenizer, AutoModel
import torch
import weaviate
from weaviate.classes.query import HybridFusion, MetadataQuery, Filter
import time
from openai import OpenAI
from typing import List, Dict
import os
from itertools import groupby
import re
from openai import OpenAI

# Initialize OpenAI and weaviate clients
os.environ["OPENAI_API_KEY"] = "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"
client_generation = OpenAI()
client_refine = OpenAI()
client_hyde = OpenAI()
client_validate = OpenAI()



class ModelManager:
    def __init__(self, model_path):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(model_path, add_pooling_layer=False)

    def get_embedding(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            embeddings = self.model(**inputs)[0][:, 0]
        return torch.nn.functional.normalize(embeddings, p=2, dim=1).squeeze(0).tolist()
model_manager = ModelManager("Models2/Embeddings/Snowflake/snowflake-arctic-embed-m")    

class AnswerValidationAgent:
    def __init__(self, max_attempts=3):
        self.max_attempts = max_attempts
        self.current_attempt = 0
        self.search_params = {'max_results': 5,  'alpha': 0.5}

    def validate_answer(self, answer, context_docs, query):
        if not context_docs:
            return False
        context_str = "\n\n".join(doc['text'] for doc in context_docs)
        # prompt = f"""
        # Legal Answer Validation - Strict Credibility Check:
        # 1. Does the answer clearly and directly reference information from the provided context documents?
        # 2. Are there any claims made that are not mentioned, implied, or supported by the documents?
        # 3. Is the answer omitting any critical or relevant context?
        # 4. Is the answer fully credible and believable based on the given documents?

        # If the answer is "there are no clear answer to the query" or not contained in the context, respond with 'invalid'.

        # Documents:
        # {context_str}

        # Query: {query}
        # Answer: {answer}

        # Respond ONLY with a single word: 'valid' or 'invalid'. Do not provide any explanation.
        # """

        prompt = f"""
        Legal Answer Validation - Strict Check:

        Evaluate the answer based on the provided documents according to these criteria:
        - Correctness: Does the answer accurately reflect information from the documents?
        - Completeness: Does the answer include all critical and relevant information?
        - Honesty: Does the answer avoid making claims not supported by the documents?

        If the answer fails any of these criteria, or if critical information is missing, or if unsupported claims are made, consider it invalid.

        Documents:
        {context_str}

        Query: {query}
        Answer: {answer}

        Respond ONLY with one word: 'valid' or 'invalid'. No explanations.
        """

        try:
            response = client_validate.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt.strip()}],
                temperature=0.0
            )
            content = response.choices[0].message.content.strip().lower()
            return 'valid' in content and 'invalid' not in content
        except Exception as e:
            return False
        

class DocumentSearcher:
    def __init__(self, host="localhost", collection_name="BAAI", alpha=0.5, context_window=1):
        self.host = host
        self.collection_name = collection_name
        self.alpha = alpha
        self.context_window = context_window

    def _connect(self):
        """Helper to connect to the Weaviate client."""
        return weaviate.connect_to_local(host=self.host)

    def search_documents(self, query_text: str, max_results: int) -> List[Dict]:
        try:
            weaviate_client = self._connect()
            collection = weaviate_client.collections.get(self.collection_name)
            response = collection.query.hybrid(
                query=query_text,
                vector=model_manager.get_embedding(query_text),
                alpha=self.alpha,
                fusion_type=HybridFusion.RELATIVE_SCORE,
                limit=max_results,
                return_properties=["text", "source", "category", "chunk_index"],
                return_metadata=MetadataQuery(score=True)
            )

            results = []
            for obj in response.objects:
                results.append({
                    "text": obj.properties["text"],
                    "source": obj.properties["source"],
                    "category": obj.properties["category"],
                    "chunk_index": obj.properties["chunk_index"],
                    "score": obj.metadata.score
                })

            # Document expansion logic
            merged_results = self._expand_documents(results)

            return merged_results
        except Exception as e:
            return {"error": "Error searching for documents", "details": str(e)}
        finally:
            weaviate_client.close()

    def _expand_documents(self, results: List[Dict]) -> List[Dict]:
        """Expand document search results by looking for adjacent chunks."""
        merged_results = []
        if results:
            expanded_chunks = self.expand_document_search(results)
            seen = {(chunk['source'], chunk['category'], chunk['chunk_index']) for chunk in results}

            for chunk in expanded_chunks:
                if (chunk['source'], chunk['category'], chunk['chunk_index']) not in seen:
                    results.append({
                        "text": chunk["text"],
                        "source": chunk["source"],
                        "category": chunk["category"],
                        "chunk_index": chunk["chunk_index"],
                        "score": 0.4})

            # Custom order for category
            category_order = {
                'Introduction': 0,
                'Preamble': 1,
                'Operative': 2,
                'Signature': 3,
                'Uncategorized': 4
            }

            # Keep highest-scoring unique (source, category, chunk_index)
            best_results = {}
            for item in results:
                key = (item['source'], item['category'], item['chunk_index'])
                if key not in best_results or item['score'] > best_results[key]['score']:
                    best_results[key] = item

            # Sort
            sorted_items = sorted(
                best_results.values(),
                key=lambda x: (
                    x['source'],
                    category_order.get(x['category'], float('inf')),
                    x['chunk_index']
                )
            )

            # Group and merge consecutive chunks
            for (source, category), group in groupby(sorted_items, key=lambda x: (x['source'], x['category'])):
                group = list(group)
                group.sort(key=lambda x: x['chunk_index'])

                merged = [group[0]]
                for item in group[1:]:
                    last = merged[-1]
                    if item['chunk_index'] == last['chunk_index'] + 1:
                        last['text'] += item['text']
                        last['score'] = max(last['score'], item['score'])
                    else:
                        merged.append(item)
                merged_results.extend(merged)

        return merged_results

    def expand_document_search(self, initial_results: List[Dict]) -> List[Dict]:
        """Expand search results by fetching adjacent chunks."""
        expanded_chunks = []
        doc_sources = set()

        # Get adjacent chunks from the same documents
        for chunk in reversed(initial_results):
            doc_sources.add((chunk['source'], chunk['category'], chunk['chunk_index']))

        try:
            weaviate_client = self._connect()
            collection = weaviate_client.collections.get(self.collection_name)
            for source, category, index in doc_sources:
                filters = (
                    Filter.by_property("source").equal(source)
                    & Filter.by_property("category").equal(category)
                    & Filter.by_property("chunk_index").greater_than(index - self.context_window - 1)
                    & Filter.by_property("chunk_index").less_than(index + self.context_window + 1)
                )

                response = collection.query.fetch_objects(
                    filters=filters,
                    return_properties=["text", "source", "category", "chunk_index"]
                )

                for obj in response.objects:
                    expanded_chunks.append({
                        "text": obj.properties["text"],
                        "source": obj.properties["source"],
                        "category": obj.properties["category"],
                        "chunk_index": obj.properties["chunk_index"],
                        "score": 0.4
                    })
            return expanded_chunks
        except Exception as e:
            return {"error": "Error expanding document search", "details": str(e)}
        finally:
            weaviate_client.close()


def generate_hypothetical_document(query: str) -> str:
    prompt_template = f"""
    You are a legal research assistant. Given a legal query, your task is to hypothesize what a relevant legal document might say in response, based on legal knowledge and common ordinance/resolution structures.

    The hypothetical document should:
    - Be plausible and consistent with actual local ordinances or legislative resolutions.
    - Mention specific topics or legal actions relevant to the query.
    - Be structured formally, like a section or excerpt from a resolution or ordinance.
    - Avoid repetition of the query itself—focus on what the document would actually contain.
    - If the response is in a language other than English, translate it into English before presenting the final output.

    Query:
    "{query}"

    Generate a hypothetical legal document (in 1–2 paragraphs) that could plausibly be retrieved in response to this query.
    """

    try:
        response = client_hyde.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a legal assistant."},
                {"role": "user", "content": prompt_template}
            ],
            temperature=0.7,
            max_tokens=512
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return "Error generating hypothetical document" 


def generate_response(query, context_docs, context_sources):
    context_str = "\n\n".join([f"Document {index + 1}:\n{doc['text']}" for index, doc in enumerate(context_docs)])
    prompt = f"""
    You are a legal AI assistant helping users find information from ordinances and resolutions. 
    Answer the query **strictly using the provided context below**. 
    Do NOT make up information or guess. If the answer is not explicitly stated in the context, respond with: 
    "Based on the available documents, there is no clear answer to the query."

    After generating the answer, also indicate the document number(s) that were used to generate your answer (e.g., Document 1, Document 3). 
    If you answered "Based on the available documents, there is no clear answer to the query.", do not include any document numbers.

    Query: {query}

    Context: {context_str if context_docs else 'No relevant documents found.'}
    """
    
    try:
        response = client_generation.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt.strip()}],
            temperature=0.1
        )
        
        generated_answer = response.choices[0].message.content.strip()

        doc_numbers = re.findall(r"document\s+(\d+)", generated_answer, re.IGNORECASE)
        relevant_sources = [context_sources[int(num) - 1] for num in doc_numbers if num.isdigit()]
        relevant_contexts = [context_docs[int(num) - 1] for num in doc_numbers if num.isdigit()]

        generated_answer = re.sub(
            r"(\(?\s*(See\s+)?(Sources?:\s*)?(Document\s+\d+[,\s]*)+(and\s+)?(Document\s+\d+)?\s*\)?)", 
            "", 
            generated_answer, 
            flags=re.IGNORECASE
        ).strip()
        return generated_answer, relevant_sources, relevant_contexts
    except Exception as e:
        return "Error generating response", [], []
    

def refine_query_with_history(new_query, chat_history):
    context_messages = []
    for msg in reversed(chat_history):
        if msg["role"] == "user":
            context_messages.insert(0, f"User: {msg['content']}")
        elif msg["role"] == "assistant":
            context_messages.insert(0, f"Assistant: {msg['content']}")
        if len(context_messages) >= 6: 
            break
    
    context_str = "\n".join(context_messages[-6:]) if context_messages else "No previous context"
    
    prompt = f"""
    You're a legal query refiner helping create effective search queries. Consider both:
    1. The new user query
    2. Relevant context from chat history (if applicable)

    Your task:
    - If the input is a valid legal-related query, refine it into a standalone, clear, **affirmative sentence** (not a question), in English, suitable for document search.
    - If the input is too vague or conversational (e.g., "ok", "thanks"), respond with a blank.

    Chat History (most recent first): {context_str}

    New Query: {new_query}

    Refined Search Query (respond ONLY with the refined query in ENGLISH, or leave it blank):
    """

    try:
        response = client_refine.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt.strip()}],
            temperature=0.3,
            max_tokens=100
        )
        return response.choices[0].message.content.strip()
    except:
        return new_query

def stream_response(text):
    for chunk in text.split():
        yield chunk + " "
        time.sleep(0.03)

# Streamlit UI with chat history management
def main():
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

    # Main chat interface
    current_chat = st.session_state.chats[st.session_state.current_chat]
    
    # Display messages for current chat
    for message in current_chat["messages"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant" and message.get("sources"):
                formatted_sources = [f"{src}"for src in message['sources']]
                st.markdown(f"**Sources:** {', '.join(formatted_sources)}")

    # Handle new input
    if user_input := st.chat_input("Ask about local ordinances and resolutions:"):
        # Add user message to current chat
        current_chat["messages"].append({"role": "user", "content": user_input})
        
        with st.chat_message("user"):
            st.markdown(user_input)

        # Process query with fresh agent for each input
        agent = AnswerValidationAgent()
        final_response = ""
        all_sources = set()

        # Get current chat history for context
        chat_history = current_chat["messages"]
        refined_query = refine_query_with_history(user_input, chat_history)
        if not refined_query.strip():  # Checks if the refined query is empty or just spaces
            final_response = "Please provide your question." 
        else:
            while agent.current_attempt < agent.max_attempts:
                if agent.current_attempt > 0:
                    refined_query = generate_hypothetical_document(refined_query)
                print(f"Attempt {agent.current_attempt + 1} with query: {refined_query}")
                searcher = DocumentSearcher(host="localhost", collection_name="BAAI", alpha=0.5)
                context_docs = searcher.search_documents(refined_query, max_results=5)
                sources = [(doc["source"], doc["category"], doc["chunk_index"]) for doc in context_docs]
                print(f"Initial sources: {sources}")
                response, relevant_sources, relevant_contexts = generate_response(user_input, context_docs, sources)
                print(f"Relevant sources: {relevant_sources}")
                if agent.validate_answer(response, relevant_contexts, user_input):
                    final_response = response
                    all_sources.update(relevant_sources)
                    break
                    
                agent.current_attempt += 1

            # Fallback response
            if not final_response:
                final_response = "No relevant documents were found in the database to answer your question. Please try rephrasing or clarifying your query."

            # Add assistant response to current chat
            current_chat["messages"].append({
                "role": "assistant",
                "query": user_input,
                "content": final_response,
                "sources": list({f"{src}" for src, cat, chunk in all_sources})
            })

        # Display assistant response
        with st.chat_message("assistant"):
            st.write(final_response)
            if all_sources:
                st.markdown(f"\n**Verified Sources:** {', '.join({src for src, cat, chunk in all_sources})}")

if __name__ == "__main__":
    main()