# inference.py
import os
import json
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from colbert.infra import Run, RunConfig
from colbert.modeling.tokenization import QueryTokenizer, DocTokenizer
from colbert.modeling.colbert import ColBERT
from colbert.utils.utils import print_message

class RerankerModel:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.models = {}
        self.tokenizers = {}
        
    def load_bge_reranker(self):
        model_name = "BAAI/bge-reranker-large"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        model.to(self.device)
        return model, tokenizer
    
    def load_colbert(self):
        with Run().context(RunConfig(nranks=1, rank=0)):
            config = {}  # Add any specific ColBERT config here
            model = ColBERT(config)
            query_tokenizer = QueryTokenizer(config)
            doc_tokenizer = DocTokenizer(config)
        model.to(self.device)
        return model, (query_tokenizer, doc_tokenizer)
    
    def load_model(self, model_dir, model_name):
        """Load model based on name parameter"""
        if model_name not in self.models:
            if model_name == "bge":
                model, tokenizer = self.load_bge_reranker()
                self.models[model_name] = model
                self.tokenizers[model_name] = tokenizer
            elif model_name == "colbert":
                model, tokenizers = self.load_colbert()
                self.models[model_name] = model
                self.tokenizers[model_name] = tokenizers
            else:
                raise ValueError(f"Unknown model: {model_name}")
        
        return self.models[model_name], self.tokenizers[model_name]

    def predict_bge(self, queries, passages):
        """Rerank using BGE reranker"""
        model = self.models["bge"]
        tokenizer = self.tokenizers["bge"]
        
        pairs = [[query, passage] for query, passage in zip(queries, passages)]
        features = tokenizer(
            pairs,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=512
        ).to(self.device)
        
        with torch.no_grad():
            scores = model(**features).logits.squeeze()
            
        return scores.cpu().tolist()

    def predict_colbert(self, queries, passages):
        """Rerank using ColBERT"""
        model = self.models["colbert"]
        query_tokenizer, doc_tokenizer = self.tokenizers["colbert"]
        
        # Process queries and documents
        Q = query_tokenizer.tensorize(queries)
        D = doc_tokenizer.tensorize(passages)
        
        # Get scores
        with torch.no_grad():
            scores = model(Q, D).cpu().tolist()
            
        return scores

    def predict_fn(self, input_data, model_name):
        """Main prediction function"""
        input_data = json.loads(input_data)
        queries = input_data["queries"]
        passages = input_data["passages"]
        
        if model_name == "bge":
            scores = self.predict_bge(queries, passages)
        elif model_name == "colbert":
            scores = self.predict_colbert(queries, passages)
        else:
            raise ValueError(f"Unknown model: {model_name}")
            
        return {"scores": scores}

# Create model instance
model = RerankerModel()

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
    """Prediction function - required by SageMaker"""
    # Extract model name from the request
    model_name = json.loads(input_data).get("model_name", "bge")
    return model.predict_fn(input_data, model_name)

def output_fn(prediction_output, accept):
    """Format output data - required by SageMaker"""
    if accept == 'application/json':
        return json.dumps(prediction_output)
    raise ValueError(f"Unsupported accept type: {accept}")