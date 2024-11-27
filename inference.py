# inference.py
import os
import json
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoModel
import numpy as np

class RankerModel:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.models = {}
        self.tokenizers = {}
        
    def load_bge(self):
        """Load BGE reranker model"""
        model_name = "BAAI/bge-reranker-large"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        model.eval()
        model.to(self.device)
        return model, tokenizer

    def load_colbert(self):
        """Load ColBERT model"""
        model_name = "colbert-ir/colbertv2.0"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        model.eval()
        model.to(self.device)
        return model, tokenizer

    def load_model(self, model_dir, model_name):
        """Load model based on name parameter"""
        if model_name not in self.models:
            if model_name == "bge":
                model, tokenizer = self.load_bge()
            elif model_name == "colbert":
                model, tokenizer = self.load_colbert()
            else:
                raise ValueError(f"Unknown model: {model_name}")
            
            self.models[model_name] = model
            self.tokenizers[model_name] = tokenizer
        
        return self.models[model_name], self.tokenizers[model_name]

    def predict_bge(self, queries, passages):
        """Get BGE scores"""
        model = self.models["bge"]
        tokenizer = self.tokenizers["bge"]
        
        all_scores = []
        for query in queries:
            query_scores = []
            for passage in passages:
                with torch.no_grad():
                    inputs = tokenizer(
                        [[query, passage]], 
                        padding=True, 
                        truncation=True, 
                        return_tensors='pt', 
                        max_length=512
                    ).to(self.device)
                    score = model(**inputs, return_dict=True).logits.view(-1,).float()
                    query_scores.append(score.item())
            all_scores.append(query_scores)
        
        return self.normalize_scores(all_scores)

    def predict_colbert(self, queries, passages):
        """Get ColBERT scores"""
        model = self.models["colbert"]
        tokenizer = self.tokenizers["colbert"]
        
        all_scores = []
        for query in queries:
            query_scores = []
            for passage in passages:
                with torch.no_grad():
                    query_inputs = tokenizer(
                        query,
                        padding=True,
                        truncation=True,
                        return_tensors='pt',
                        max_length=32
                    ).to(self.device)
                    
                    passage_inputs = tokenizer(
                        passage,
                        padding=True,
                        truncation=True,
                        return_tensors='pt',
                        max_length=180
                    ).to(self.device)

                    query_embedding = model(**query_inputs).last_hidden_state
                    passage_embedding = model(**passage_inputs).last_hidden_state
                    
                    similarity = torch.matmul(query_embedding, passage_embedding.transpose(-1, -2))
                    score = similarity.max(dim=-1)[0].sum().item()
                    query_scores.append(score)
            all_scores.append(query_scores)
            
        return self.normalize_scores(all_scores)

    def normalize_scores(self, scores):
        """Normalize scores to 0-100 range"""
        scores_flat = [score for sublist in scores for score in sublist]
        scores_array = np.array(scores_flat)
        
        if np.max(scores_array) == np.min(scores_array):
            normalized = np.zeros_like(scores_array)
        else:
            normalized = 100 * (scores_array - np.min(scores_array)) / (np.max(scores_array) - np.min(scores_array))
        
        # Reshape back to original structure
        normalized_scores = []
        idx = 0
        for sublist in scores:
            normalized_scores.append(normalized[idx:idx+len(sublist)].tolist())
            idx += len(sublist)
            
        return normalized_scores

# Create model instance
model = RankerModel()

def model_fn(model_dir):
    """Load model - required by SageMaker"""
    return model

def input_fn(request_body, request_content_type):
    """Parse input data - required by SageMaker"""
    if request_content_type == 'application/json':
        return request_body
    else:
        raise ValueError(f"Unsupported content type: {request_content_type}")

def predict_fn(input_data, model):
    """
    Prediction function - required by SageMaker
    Expected input format:
    {
        "model_name": "bge" or "colbert",
        "queries": ["query1", "query2", ...],
        "passages": ["passage1", "passage2", ...]
    }
    """
    input_object = json.loads(input_data)
    model_name = input_object.get("model_name", "bge")
    queries = input_object["queries"]
    passages = input_object["passages"]
    
    # Load model if not already loaded
    model.load_model(None, model_name)
    
    # Get predictions based on model type
    if model_name == "bge":
        scores = model.predict_bge(queries, passages)
    elif model_name == "colbert":
        scores = model.predict_colbert(queries, passages)
    else:
        raise ValueError(f"Unknown model: {model_name}")
        
    return {
        "scores": scores,
        "metadata": {
            "model_used": model_name,
            "num_queries": len(queries),
            "num_passages": len(passages)
        }
    }

def output_fn(prediction_output, accept):
    """Format output data - required by SageMaker"""
    if accept == 'application/json':
        return json.dumps(prediction_output)
    raise ValueError(f"Unsupported accept type: {accept}")