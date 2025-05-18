from transformers import AutoTokenizer, AutoModel
import torch

class ModelManager:
    def __init__(self, model_path):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(model_path, add_pooling_layer=True)

    def get_embedding(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            embeddings = self.model(**inputs)[0][:, 0]
        return torch.nn.functional.normalize(embeddings, p=2, dim=1).squeeze(0).tolist()