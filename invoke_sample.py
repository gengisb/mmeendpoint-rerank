import boto3
import json

runtime = boto3.client('sagemaker-runtime')

# Example input
payload = {
    "model_name": "bge",  # or "colbert"
    "queries": ["what is machine learning?"],
    "passages": [
        "Machine learning is a subset of artificial intelligence...",
        "The weather is nice today."
    ]
}

response = runtime.invoke_endpoint(
    EndpointName=endpoint_name,
    ContentType='application/json',
    Body=json.dumps(payload)
)

result = json.loads(response['Body'].read().decode())
print(result)