# chatapp/services.py

import os
import weaviate
from transformers import AutoTokenizer, AutoModel
from openai import OpenAI

# OpenAI client setup
os.environ["OPENAI_API_KEY"] = "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"

client_generation = OpenAI()
client_refine = OpenAI()
client_hyde = OpenAI()
client_validate = OpenAI()

# Weaviate connection
def get_weaviate_client():
    return weaviate.connect_to_local(host="localhost")  # or your container host if Dockerized

# # Load embedding model (only once, cached)
# def load_models():
#     embed_model_name = r"/home/ubuntu/Models3/Embeddings/Snowflake/snowflake-arctic-embed-m"
#     embed_tokenizer = AutoTokenizer.from_pretrained(embed_model_name)
#     embed_model = AutoModel.from_pretrained(embed_model_name, add_pooling_layer=False)
#     return embed_tokenizer, embed_model
