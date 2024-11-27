from sagemaker.huggingface import HuggingFaceModel
import sagemaker

role = sagemaker.get_execution_role()

# Create HuggingFace Model
huggingface_model = HuggingFaceModel(
    model_data=None,  # No model artifact since we're downloading in code
    role=role,
    transformers_version="4.28",
    pytorch_version="2.0",
    py_version="py310",
    entry_point="inference.py",
    source_dir="./",
    instance_type="ml.g4dn.xlarge",  # Use GPU instance
)

# Deploy multi-model endpoint
endpoint_name = "reranker-endpoint"
huggingface_model.deploy(
    initial_instance_count=1,
    instance_type="ml.g4dn.xlarge",
    endpoint_name=endpoint_name
)