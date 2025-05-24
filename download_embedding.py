from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-base-en-v1.5")
model = AutoModel.from_pretrained("BAAI/bge-base-en-v1.5")

save_path = "./embeddings"
tokenizer.save_pretrained(save_path)
model.save_pretrained(save_path)