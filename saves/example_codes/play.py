import ollama

client = ollama.Client(host="http://atlas1api.eurecom.fr:8019")
response = client.chat(model="mistral-nemo:12b-instruct-2407-fp16",
                       messages=[{"role": "user", "content": "Why is the sky blue?"}])
print(response["message"]["content"])

question = response["message"]["content"]