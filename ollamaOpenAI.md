# Using Ollama API Instead of OpenAI
If localhost does not work use 127.0.0.1 

## Basic URL Replacement
- OpenAI API: `https://api.openai.com/v1/chat/completions`
- Ollama API: `http://localhost:11434/api/chat` (for local server)
- Ollama API: `http://<your-server-ip>:11434/api/chat` (for remote server)

## Example with cURL

```bash
# OpenAI style request
curl https://api.openai.com/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -d '{
    "model": "gpt-3.5-turbo",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'

# Equivalent Ollama request
curl http://localhost:11434/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-r1",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

## Example with Python

```python
# OpenAI client
import openai
openai.api_key = "your-api-key"
response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": "Hello!"}]
)

# Ollama client
import requests
response = requests.post(
    "http://localhost:11434/api/chat",
    json={
        "model": "deepseek-r1",
        "messages": [{"role": "user", "content": "Hello!"}]
    }
)
```

## Using with Libraries That Support Base URL Override

Many libraries that wrap OpenAI's API support changing the base URL:

```python
# Using OpenAI's official Python client with Ollama
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:11434/api",  # Use your Ollama server URL
    api_key="ollama",  # Can be any string as Ollama doesn't check API keys
)

response = client.chat.completions.create(
    model="deepseek-r1",  # Your model name in Ollama
    messages=[{"role": "user", "content": "Hello!"}]
)
```
