import requests
import json

# with open("../../resources/openai.key", 'r') as f:
#     key = f.readlines()[0][:-1]

key = "sk-hN7u5Yuz9Z7RQe4OA5F61958E006480d9dE11a0a9e0d18Ac"

def embedding_retriever(term):
    # Set up the API endpoint URL and request headers
    # url = "https://api.openai.com/v1/embeddings"
    url = "https://xiaoai.plus/v1/embeddings"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {key}"
    }

    # Set up the request payload with the text string to embed and the model to use
    payload = {
        "input": term,
        "model": "text-embedding-ada-002"
    }

    # Send the request and retrieve the response
    response = requests.post(url, headers=headers, data=json.dumps(payload))
    # print(response.json())
    # Extract the text embeddings from the response JSON
    embedding = response.json()["data"][0]['embedding']

    return embedding