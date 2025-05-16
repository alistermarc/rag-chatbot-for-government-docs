from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from django.shortcuts import render
from rest_framework.parsers import MultiPartParser, FormParser
import os
import weaviate
from .textract import TextractProcessor, DocumentPreprocessor, DocumentIndexer
from .chat_agent import AnswerValidationAgent, DocumentSearcher, refine_query_with_history, generate_hypothetical_document, generate_response, classify_if_retrieval_needed, conversation_without_retrieval
# from .services import client_generation, client_refine, get_weaviate_client, load_models
from transformers import AutoTokenizer, AutoModel
import torch
# embed_tokenizer, embed_model = load_models()

class ModelManager:
    def __init__(self, model_path):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(model_path, add_pooling_layer=True)

    def get_embedding(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            embeddings = self.model(**inputs)[0][:, 0]
        return torch.nn.functional.normalize(embeddings, p=2, dim=1).squeeze(0).tolist()
model_manager = ModelManager(r"/home/ubuntu/Capstone-Project/embedding/")   


class ChatAPIView(APIView):
    def post(self, request):
        user_input = request.data.get("message")
        chat_history = request.data.get("history", [])

        agent = AnswerValidationAgent()
        final_response = ""
        all_sources = set()

        if not classify_if_retrieval_needed(user_input, chat_history):
            final_response = conversation_without_retrieval(user_input, chat_history)
        else:
            refined_query = refine_query_with_history(user_input, chat_history)
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
            "sources": list({f"{src}" for src, cat, chunk in all_sources})
        }, status=status.HTTP_200_OK)


class FileUploadAPIView(APIView):
    parser_classes = [MultiPartParser, FormParser]

    def post(self, request):
        uploaded_file = request.FILES.get("file")

        if not uploaded_file:
            return Response({"error": "No file uploaded."}, status=status.HTTP_400_BAD_REQUEST)

        if not uploaded_file.name.lower().endswith(".pdf"):
            return Response({"error": "Only PDF files are supported."}, status=status.HTTP_400_BAD_REQUEST)

        file_name = uploaded_file.name
        base_file_name = file_name[:-4]
        save_dir = "New Documents"
        os.makedirs(save_dir, exist_ok=True)
        file_path = os.path.join(save_dir, file_name)

        # Save the file
        with open(file_path, "wb") as f:
            f.write(uploaded_file.read())

        # Check if already indexed
        client = weaviate.connect_to_local()
        collection = client.collections.get("BAAI")
        existing_sources = [obj.properties.get("source", "No source") for obj in collection.iterator(return_properties=["source"])]

        if base_file_name in existing_sources:
            return Response({"message": f"'{file_name}' already exists in the database."}, status=status.HTTP_200_OK)

        # Run Textract
        output_folder = os.path.join(save_dir, base_file_name)
        layout_csv_path = os.path.join(output_folder, "layout.csv")

        processor = TextractProcessor(base_dir=save_dir)
        processor.process_pdf(file_path)

        if not os.path.exists(layout_csv_path):
            return Response({"error": "Textract processing failed."}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        # Preprocess and index
        documentpreprocessor = DocumentPreprocessor()
        documentpreprocessor.summarize(output_folder)

        processtodatabase = DocumentIndexer(model_manager)
        processtodatabase.process_file_to_db(output_folder)

        return Response({
            "message": f"File '{file_name}' uploaded and processed successfully.",
            "source": base_file_name
        }, status=status.HTTP_201_CREATED)


class CheckAPIView(APIView):
    def get(self, request):
        return Response({"status": "Server is running"}, status=status.HTTP_200_OK)
