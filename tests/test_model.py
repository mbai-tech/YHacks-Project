import urllib.request, json

data = json.dumps({
    "model": "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit",
    "messages": [{"role": "user", "content": "What is the highest-risk facility in Hartford County, CT?"}],
    "max_tokens": 300
}).encode()

req = urllib.request.Request(
    "http://127.0.0.1:11434/v1/chat/completions",
    data=data,
    headers={"Content-Type": "application/json"}
)
with urllib.request.urlopen(req) as r:
    result = json.loads(r.read())
    print(result["choices"][0]["message"]["content"])
