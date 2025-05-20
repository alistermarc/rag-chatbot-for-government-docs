from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from django.shortcuts import render
from rest_framework.parsers import MultiPartParser, FormParser
import os
import weaviate
from .textract import TextractProcessor, DocumentPreprocessor, DocumentIndexer
from .chat_agent import RetrievalDecisionModule, QueryTransformationModule, DocumentRetrievalModule, ResponseGeneratorModule, AnswerValidationAgent


class ChatAPIView(APIView):
    def post(self, request):
        user_input = request.data.get("message")
        chat_history = request.data.get("history", [])

        context_messages = []
        for msg in reversed(chat_history):
            if msg["role"] == "user":
                context_messages.insert(0, f"User: {msg['content']}")
            elif msg["role"] == "assistant":
                context_messages.insert(0, f"Assistant: {msg['content']}")
            if len(context_messages) >= 6:
                break
        context_str = "\n".join(context_messages) if context_messages else "No previous context"

        retrieval_decision_module = RetrievalDecisionModule()
        query_transformation_module = QueryTransformationModule()
        response_generator_module = ResponseGeneratorModule()

        final_response = ""
        all_sources = set()

        if not retrieval_decision_module.classify_if_retrieval_needed(user_input, context_str):
            final_response = response_generator_module.conversation_without_retrieval(user_input, context_str)
        else:
            refined_query = query_transformation_module.refine_query_with_history(user_input, context_str)

            with DocumentRetrievalModule(host="localhost", collection_name="BAAI", alpha=0.5) as searcher:
                agent = AnswerValidationAgent()
                while agent.current_attempt < agent.max_attempts:
                    if agent.current_attempt > 0:
                        refined_query = query_transformation_module.generate_hypothetical_document(refined_query)

                    print(f"Attempt {agent.current_attempt + 1} with query: {refined_query}")
                    
                    context_docs = searcher.search_documents(refined_query, max_results=7)
                    sources = [(doc["source"], doc["category"], doc["chunk_index"]) for doc in context_docs]
                    print(f"Initial sources: {sources}")

                    response, relevant_sources, relevant_contexts = response_generator_module.generate_response(user_input, context_docs, sources)
                    print(f"Relevant sources: {relevant_sources}")

                    if agent.validate_answer(response, relevant_contexts, user_input) and relevant_sources:
                        final_response = response
                        all_sources.update(relevant_sources)
                        break

                    agent.current_attempt += 1
                
            # Fallback response
            if not final_response:
                final_response = "No relevant documents were found in the database to answer your question. Please try rephrasing or clarifying your query."

        return Response({
            "answer": final_response,
            "sources": list(set({f"{src}" for src, cat, chunk in all_sources}))
        }, status=status.HTTP_200_OK)


class FileUploadAPIView(APIView):
    parser_classes = [MultiPartParser, FormParser]

    def post(self, request):
        uploaded_files = request.FILES.getlist("file")

        if not uploaded_files:
            return Response({"error": "No files uploaded."}, status=status.HTTP_400_BAD_REQUEST)

        results = []
        client = weaviate.connect_to_local()

        try:
            collection = client.collections.get("BAAI")
            existing_sources = [
                obj.properties.get("source", "No source")
                for obj in collection.iterator(return_properties=["source"])
            ]

            for uploaded_file in uploaded_files:
                file_name = uploaded_file.name

                if '-S-' in file_name:
                    parts = file_name.split('-S-')
                    file_name = f"{parts[0]}, S-{parts[1]}"

                if not file_name.lower().endswith(".pdf"):
                    results.append({
                        "file": file_name,
                        "status": "failed",
                        "message": "Only PDF files are supported."
                    })
                    continue

                base_file_name = file_name[:-4]

                if base_file_name in existing_sources:
                    results.append({
                        "file": file_name,
                        "status": "failed",
                        "message": "File already exists in the database."
                    })
                    continue

                save_dir = "New Documents"
                os.makedirs(save_dir, exist_ok=True)
                file_path = os.path.join(save_dir, file_name)

                with open(file_path, "wb") as f:
                    f.write(uploaded_file.read())

                output_folder = os.path.join(save_dir, base_file_name)
                layout_csv_path = os.path.join(output_folder, "layout.csv")

                processor = TextractProcessor(base_dir=save_dir)
                processor.process_pdf(file_path)

                if not os.path.exists(layout_csv_path):
                    results.append({
                        "file": file_name,
                        "status": "failed",
                        "message": "Textract processing failed."
                    })
                    continue

                documentpreprocessor = DocumentPreprocessor()
                documentpreprocessor.summarize(output_folder)

                processtodatabase = DocumentIndexer()
                processtodatabase.process_file_to_db(output_folder)

                results.append({
                    "file": file_name,
                    "status": "success",
                    "message": "Uploaded and processed successfully.",
                    "source": base_file_name
                })
            print(f"Results: {results}")
        finally:
            client.close()

        return Response({"results": results}, status=status.HTTP_207_MULTI_STATUS)


class CheckAPIView(APIView):
    def get(self, request):
        return Response({"status": "Server is running"}, status=status.HTTP_200_OK)