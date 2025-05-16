from transformers import AutoTokenizer, AutoModel

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-base-en-v1.5")
model = AutoModel.from_pretrained("BAAI/bge-base-en-v1.5")

# Save them to a local directory (e.g., './bge-base-en-v1.5')
save_path = "embedding"
tokenizer.save_pretrained(save_path)
model.save_pretrained(save_path)
