# test_rerankers.py
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoModel
from tabulate import tabulate
import numpy as np
import torch.nn.functional as F

def test_bge_reranker(queries, passages):
    print("\nTesting BGE Reranker...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading BGE model and tokenizer...")
    model_name = "BAAI/bge-reranker-large"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.eval()
    model.to(device)

    scores = []
    print("\nComputing BGE scores...")
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
                ).to(device)
                score = model(**inputs, return_dict=True).logits.view(-1,).float()
                # Keep raw logits for now, we'll normalize all scores together later
                query_scores.append(score.item())
        scores.append(query_scores)
    return scores

def test_colbert(queries, passages):
    print("\nTesting ColBERT...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading ColBERT model and tokenizer from HuggingFace...")
    model_name = "colbert-ir/colbertv2.0"
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    base_model = AutoModel.from_pretrained(model_name)
    base_model.to(device)
    base_model.eval()

    scores = []
    print("\nComputing ColBERT scores...")
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
                ).to(device)
                
                passage_inputs = tokenizer(
                    passage,
                    padding=True,
                    truncation=True,
                    return_tensors='pt',
                    max_length=180
                ).to(device)

                query_embedding = base_model(**query_inputs).last_hidden_state
                passage_embedding = base_model(**passage_inputs).last_hidden_state

                # Calculate maxsim score
                similarity = torch.matmul(query_embedding, passage_embedding.transpose(-1, -2))
                score = similarity.max(dim=-1)[0].sum().item()
                query_scores.append(score)
        scores.append(query_scores)
    return scores

def normalize_scores(bge_scores, colbert_scores):
    """Normalize both sets of scores to be roughly in the same range"""
    
    # Flatten score lists
    bge_flat = [score for sublist in bge_scores for score in sublist]
    colbert_flat = [score for sublist in colbert_scores for score in sublist]
    
    # Convert to numpy arrays for easier manipulation
    bge_array = np.array(bge_flat)
    colbert_array = np.array(colbert_flat)
    
    # Min-max normalization to [0, 100] range for both
    def min_max_normalize(arr):
        return 100 * (arr - np.min(arr)) / (np.max(arr) - np.min(arr))
    
    bge_normalized = min_max_normalize(bge_array)
    colbert_normalized = min_max_normalize(colbert_array)
    
    # Reshape back to original structure
    bge_norm_scores = []
    colbert_norm_scores = []
    idx = 0
    for sublist in bge_scores:
        bge_norm_scores.append(bge_normalized[idx:idx+len(sublist)].tolist())
        colbert_norm_scores.append(colbert_normalized[idx:idx+len(sublist)].tolist())
        idx += len(sublist)
        
    return bge_norm_scores, colbert_norm_scores

def display_comparison(queries, passages, bge_scores, colbert_scores):
    print("\n🔍 Model Comparison Results (Scores normalized to 0-100 range)")
    print("-----------------------------------------------------------")
    
    # Create headers for the table
    headers = ["Query", "Passage", "BGE Score", "ColBERT Score", "Difference"]
    
    # Prepare table rows
    rows = []
    for q_idx, query in enumerate(queries):
        for p_idx, passage in enumerate(passages):
            display_passage = passage[:50] + "..." if len(passage) > 50 else passage
            bge_score = bge_scores[q_idx][p_idx]
            colbert_score = colbert_scores[q_idx][p_idx]
            
            rows.append([
                query,
                display_passage,
                f"{bge_score:.1f}",
                f"{colbert_score:.1f}",
                f"{abs(bge_score - colbert_score):.1f}"
            ])
    
    print(tabulate(rows, headers=headers, tablefmt="grid"))

def main():
    print("Starting reranker tests...")
    
    queries = [
        "what is machine learning?",
        "what is deep learning?"
    ]
    
    passages = [
        "Machine learning is a subset of artificial intelligence that focuses on data and algorithms to imitate how humans learn.",
        "Deep learning is a type of machine learning based on artificial neural networks.",
        "The weather is nice today.",
    ]
    
    try:
        bge_scores = test_bge_reranker(queries, passages)
        colbert_scores = test_colbert(queries, passages)
        
        # Normalize scores to same range
        bge_norm, colbert_norm = normalize_scores(bge_scores, colbert_scores)
        
        # Display normalized scores
        display_comparison(queries, passages, bge_norm, colbert_norm)
        
    except Exception as e:
        print(f"Error during testing: {str(e)}")
        raise e

if __name__ == "__main__":
    main()