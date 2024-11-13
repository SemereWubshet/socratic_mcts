import ollama
from openai import OpenAI



with open('../templates/query_llm.txt', 'r') as f:
    query_role = f.read()

def openai_gen_soc_questions(content):
    client = OpenAI()
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": query_role},
            {"role": "user", "content": content}
        ]
    )
    questions = response.choices[0].message.content
    return questions

def ollama_gen_soc_questions(content):
    client = ollama.Client(host="http://atlas1api.eurecom.fr:8019")
    response = client.chat(model="llama3.1", messages=[{"role": "system", "content": query_role},
                                                       {"role": "user", "content": content}])
    # print(response["message"]["content"])

    questions = response["message"]["content"]
    return questions