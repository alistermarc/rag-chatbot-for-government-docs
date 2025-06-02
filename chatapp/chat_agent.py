import os
import re
from contextlib import AbstractContextManager
from itertools import groupby
from typing import Dict, List

import weaviate
from openai import OpenAI
from weaviate.classes.query import Filter, HybridFusion, MetadataQuery

from .models import ModelManager

# Initialize models 
base_dir = os.path.dirname(os.path.abspath(__file__))
embedding_path = os.path.join(base_dir, "..", "embeddings")
# For embedding model
model_manager = ModelManager(embedding_path)  
# For classification, query transformation, generation, and validation
client = OpenAI()
  

class RetrievalDecisionModule:
    def __init__(self, client=client, model="gpt-4o-mini", temperature=0.0, max_tokens=5):
        self.client = client
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def classify_if_retrieval_needed(self, user_input, context_str):
        prompt = f"""
        You are an assistant helping decide whether a user message needs document retrieval or not.

        Instructions:
        - If the user's message is purely conversational (e.g. "hi", "thanks", "ok", "that's helpful") or can be answered from previous chat messages or common knowledge (e.g. general facts), respond: **"no"**.
        - If the message requires external knowledge, document retrieval, or detailed information not provided in the chat history, respond: **"yes"**.
        - If the user's message is a simple factual question (e.g., "What is the capital of France?" or "How many days are in a week?"), respond: **"no"**.
        
        Chat History:
        {context_str}

        User Input:
        {user_input}

        Does this query need document retrieval? (Answer only "yes" or "no")
        """

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt.strip()}],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            return response.choices[0].message.content.strip().lower() == "yes"
        except Exception:
            return True


class QueryTransformationModule:
    def __init__(
        self,
        client=client,
        refine_model="gpt-4o-mini",
        hyde_model="gpt-4o-mini-search-preview",
        refine_temperature=0.3,
        hyde_temperature=0.7,
        refine_max_tokens=300,
        hyde_max_tokens=512
    ):
        self.client = client
        self.refine_model = refine_model
        self.hyde_model = hyde_model
        self.refine_temperature = refine_temperature
        self.hyde_temperature = hyde_temperature
        self.refine_max_tokens = refine_max_tokens
        self.hyde_max_tokens = hyde_max_tokens

    def refine_query_with_history(self, new_query, context_str):
        prompt = f"""
        You're legal query refiner helping create effective search queries for Quezon City documents. Consider both:
        1. The new user query
        2. Relevant context from chat history (if applicable)

        Your task:
        - Refine the query into a standalone, clear, **affirmative sentence** (not a question), in English, suitable for document search.

        Chat History (most recent first): {context_str}

        New Query: {new_query}

        Refined Search Query (respond ONLY with the refined query in ENGLISH):
        """

        try:
            response = self.client.chat.completions.create(
                model=self.refine_model,
                messages=[{"role": "user", "content": prompt.strip()}],
                temperature=self.refine_temperature,
                max_tokens=self.refine_max_tokens
            )
            return response.choices[0].message.content.strip()
        except Exception:
            return new_query

    def generate_hypothetical_document(self, query: str) -> str:
        prompt = f"""
        You are a legal research assistant. Based on the query below, generate an answer that could plausibly address it.

        - The content should be realistic and relevant.
        - Structure it formally, like a legal provision.
        - Translate to English if needed.

        Query: "{query}"

        Hypothetical legal document (1 paragraph):
        """

        try:
            response = self.client.chat.completions.create(
                model=self.hyde_model,
                messages=[
                    {"role": "user", "content": prompt.strip()}
                ],
                max_tokens=self.hyde_max_tokens
            )
            return response.choices[0].message.content.strip()
        except Exception:
            return "Error generating hypothetical document"


class DocumentRetrievalModule(AbstractContextManager):
    def __init__(self, host="localhost", collection_name="BAAI", alpha=0.5, context_window=1):
        self.host = host
        self.collection_name = collection_name
        self.alpha = alpha
        self.context_window = context_window
        self.client = None

    def __enter__(self):
        self.client = weaviate.connect_to_local(host=self.host)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.client:
            self.client.close()

    def search_documents(self, query_text: str, max_results: int) -> List[Dict]:
        try:
            self.collection = self.client.collections.get(self.collection_name)
            response = self.collection.query.hybrid(
                query=query_text,
                vector=model_manager.get_embedding(query_text),
                alpha=self.alpha,
                fusion_type=HybridFusion.RELATIVE_SCORE,
                limit=max_results,
                return_properties=["text", "source", "category", "chunk_index"],
                return_metadata=MetadataQuery(score=True)
            )

            results = []
            seen = set()

            for obj in response.objects:
                key = (obj.properties["source"], obj.properties["category"], obj.properties["chunk_index"])
                if key not in seen:
                    seen.add(key)
                    results.append({
                        "text": obj.properties["text"],
                        "source": obj.properties["source"],
                        "category": obj.properties["category"],
                        "chunk_index": obj.properties["chunk_index"],
                        "score": obj.metadata.score
                    })

            merged_results = self.expand_documents(results)
            return merged_results

        except Exception as e:
            return {"error": "Error searching for documents", "details": str(e)}

    def expand_documents(self, results: List[Dict]) -> List[Dict]:
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
                        "score": 0})

            category_order = {
                'Introduction': 0,
                'Preamble': 1,
                'Operative': 2,
                'Signature': 3,
                'Uncategorized': 4
            }

            sorted_items = sorted(
                results,
                key=lambda x: (
                    x['source'].lower(),
                    int(category_order.get(x['category'], 4)),
                    int(x['chunk_index'])
                )
            )

            for (source, category), group in groupby(sorted_items, key=lambda x: (x['source'], x['category'])):
                group = list(group)
                group.sort(key=lambda x: x['chunk_index'])

                merged = [group[0]]
                for item in group[1:]:
                    last = merged[-1]

                    if item['chunk_index'] == last['chunk_index'] + 1:
                        last['text'] += item['text']
                        last['score'] = max(last['score'], item['score'])
                        last['chunk_index'] = item['chunk_index']
                    else:
                        merged.append(item)
                merged_results.extend(merged)
        return sorted(merged_results, key=lambda x: x['score'], reverse=True)

    def expand_document_search(self, initial_results: List[Dict]) -> List[Dict]:
        expanded_chunks = []
        doc_sources = set()

        for chunk in reversed(initial_results):
            doc_sources.add((chunk['source'], chunk['category'], chunk['chunk_index']))

        try:
            for source, category, index in doc_sources:
                filters = (
                    Filter.by_property("source").equal(source)
                    & Filter.by_property("category").equal(category)
                    & Filter.by_property("chunk_index").greater_than(index - self.context_window - 1)
                    & Filter.by_property("chunk_index").less_than(index + self.context_window + 1)
                )

                response = self.collection.query.fetch_objects(
                    filters=filters,
                    return_properties=["text", "source", "category", "chunk_index"]
                )

                for obj in response.objects:
                    expanded_chunks.append({
                        "text": obj.properties["text"],
                        "source": obj.properties["source"],
                        "category": obj.properties["category"],
                        "chunk_index": obj.properties["chunk_index"],
                        "score": 0
                    })
            return expanded_chunks

        except Exception as e:
            return {"error": "Error expanding document search", "details": str(e)}


class ResponseGeneratorModule:
    def __init__(
        self,
        client=client,
        generation_model_with_retrieval="gpt-4o-mini",
        generation_model_without_retrieval="gpt-4o-mini-search-preview",
        generation_temperature=0.1,
        max_tokens=512
    ):
        self.client = client
        self.generation_model_with_retrieval = generation_model_with_retrieval
        self.generation_model_without_retrieval = generation_model_without_retrieval
        self.generation_temperature = generation_temperature
        self.max_tokens = max_tokens

    def conversation_without_retrieval(self, user_input, context_str=None):
        prompt = f"""
        You are Alex, a Quezon City Legal Provider. Answer the query using your internal knowledge or the provided conversation history if applicable. 

        Conversation history:
        {context_str if context_str else 'No previous conversation history.'}

        User query:
        {user_input}

        Please answer clearly and accurately.  
        Note: If the user query is in English, provide the answer in English. If the query is in Filipino, provide the answer in Filipino.
        """

        try:
            response = self.client.chat.completions.create(
                model=self.generation_model_without_retrieval,
                messages=[{"role": "user", "content": prompt.strip()}],
                max_tokens=self.max_tokens
            )
            return response.choices[0].message.content.strip()
        except Exception:
            return "Error generating response"


    def generate_response(self, query, context_docs, context_sources):
            
        context_str = "\n\n".join([
            f"Document {index + 1} (Source: {context_sources[index]}):\n{doc['text']}"
            for index, doc in enumerate(context_docs)
        ])
        
        prompt = f"""
        You are a legal AI assistant helping users find information from ordinances and resolutions. 
        Answer the query **strictly using the provided context below**. Give an empty answer, if answer is not in the context.
        If you use any context in your answer, you must clearly indicate which document(s) you used only at the end of your response using the format: "Document X" (e.g. Document 1, Document 2). Do not include any document numbers if response was not based in any of the documents.


        Query: {query}

        Context: {context_str if context_docs else 'No relevant documents found.'}

        Note: If the user query is in English, provide the answer in English. If the query is in Filipino, provide the answer in Filipino.
        """

        try:
            response = self.client.chat.completions.create(
                model=self.generation_model_with_retrieval,
                messages=[{"role": "user", "content": prompt.strip()}],
                temperature=self.generation_temperature,
                max_tokens=self.max_tokens
            )

            generated_answer = response.choices[0].message.content.strip()

            doc_numbers = list(set(re.findall(r"document\s+(\d+)", generated_answer, re.IGNORECASE)))
            relevant_sources = [context_sources[int(num) - 1] for num in doc_numbers if num.isdigit()]
            relevant_contexts = [context_docs[int(num) - 1] for num in doc_numbers if num.isdigit()]
            print(doc_numbers)
            generated_answer = re.sub(
                r"(\(?\s*(See\s+)?(Sources?:\s*)?(Document\s+\d+[,\s]*)+(and\s+)?(Document\s+\d+)?\s*\)?)", 
                "", 
                generated_answer, 
                flags=re.IGNORECASE
            ).strip()

            return generated_answer, relevant_sources, relevant_contexts
        except Exception:
            return "Error generating response", [], []


class AnswerValidationAgent:
    def __init__(
        self,
        client=client,
        model="gpt-4o-mini",
        temperature=0.0,
        max_tokens=512,
        max_attempts=3
    ):
        self.client = client
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_attempts = max_attempts
        self.current_attempt = 0
        self.search_params = {'max_results': 7, 'alpha': 0.5}

    def validate_answer(self, answer, context_docs, query):
        if not context_docs:
            return False

        context_str = "\n\n".join(doc['text'] for doc in context_docs)
        prompt = f"""
        Legal Answer Validation - Strict Check:

        Evaluate the answer based on the provided documents according to these criteria:
        - Correctness: Does the answer accurately reflect information from the documents?
        - Completeness: Does the answer include all critical and relevant information to the query?
        - Honesty: Does the answer avoid making claims not supported by the documents?

        If the answer fails any of these criteria, or if critical information is missing, or if unsupported claims are made, consider it invalid.

        Documents:
        {context_str}

        Query: {query}
        Answer: {answer}

        Respond ONLY with one word: 'valid' or 'invalid'. No explanations.
        """

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt.strip()}],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            content = response.choices[0].message.content.strip().lower()
            return 'valid' in content and 'invalid' not in content
        except Exception:
            return False

    def refine_answer(self, question, answer):
        prompt = f"""
        Improve the phrasing of the answer based on the question.
        Do not add or remove information. Just make it clear and well-written.

        Question: {question}
        Answer: {answer}

        Refined Answer:
        """

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt.strip()}],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            refined = response.choices[0].message.content.strip()
            return refined
        except Exception:
            return answer
