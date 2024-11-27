import boto3
import json

runtime = boto3.client('sagemaker-runtime')

# Example input data
input_data = {
    "model_name": "bge",  # or "colbert"
    "queries": ["what is machine learning?"],
    "passages": ["Machine learning is a subset of artificial intelligence..."]
}

response = runtime.invoke_endpoint(
    EndpointName=endpoint_name,
    ContentType='application/json',
    Body=json.dumps(input_data)
)

scores = json.loads(response['Body'].read().decode())
print(scores)