# in app.py
import ollama

if __name__ == "__main__":
    client = ollama.Client(host="http://atlas1api.eurecom.fr:8019")

    response = client.chat(model="llama3.1", messages=[{"role": "user", "content": "Why is the sky blue?"}])

    print(type(response["message"]["content"]))