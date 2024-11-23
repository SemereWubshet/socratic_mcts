import ollama
from openai import OpenAI



with open('templates/query_generator.txt', 'r') as f:
    query_role = f.read()

with open('templates/judge.txt', 'r') as f:
    judge_role = f.read()

with open('templates/seed.txt', 'r') as f:
    seed_role = f.read()

with open('templates/student.txt', 'r') as f:
    student_role = f.read()

with open('templates/teacher.txt', 'r') as f:
    teacher_role = f.read()

"""Query functions by Open AI"""

def openai_gen_soc_question(content):
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

def openai_gen_seed(content):
    client = OpenAI()
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": seed_role},
                  {"role": "user", "content": content}])
    seed = response.choices[0].message.content
    return seed

def openai_gen_student_response(seed, history):
    client = OpenAI()
    content = "Topic: " + seed + "\n Conversation History: " + history # if history else None
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": student_role},
                  {"role": "user", "content": content}])
    student_response = response.choices[0].message.content
    return student_response

def openai_gen_teacher_response(content):
    client = OpenAI()
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": teacher_role},
                  {"role": "user", "content": content}])
    teacher_response = response.choices[0].message.content
    return teacher_response

def openai_gen_judge(text_chunk, seed, history):
    client = OpenAI()
    content = ("Overall topic: " + text_chunk +
               "\n Seed question: " + seed +
               "\n Conversation History: " + history)
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": judge_role},
                  {"role": "user", "content": content}])
    judge_response = response.choices[0].message.content
    return judge_response


"""Query functions by Ollama"""

def ollama_gen_soc_question(content):
    client = ollama.Client(host="http://atlas1api.eurecom.fr:8019")
    response = client.chat(model="llama3.1", messages=[{"role": "system", "content": query_role},
                                                       {"role": "user", "content": content}])
    questions = response["message"]["content"]
    return questions

def ollama_gen_seed(content):
    client = ollama.Client(host="http://atlas1api.eurecom.fr:8019")
    response = client.chat(model="llama3.1", messages=[{"role": "system", "content": seed_role},
                                                       {"role": "user", "content": content}])
    seed = response["message"]["content"]
    return seed

def ollama_gen_student_response(seed, history):
    content = "Topic: " + seed + "\n Conversation History: " + history # if history else None
    client = ollama.Client(host="http://atlas1api.eurecom.fr:8019")
    response = client.chat(model="llama3.1", messages=[{"role": "system", "content": student_role},
                                                       {"role": "user", "content": content}])
    student_response = response["message"]["content"]
    return student_response

def ollama_gen_teacher_response(content):
    client = ollama.Client(host="http://atlas1api.eurecom.fr:8019")
    response = client.chat(model="llama3.1", messages=[{"role": "system", "content": teacher_role},
                                                       {"role": "user", "content": content}])
    teacher_response = response["message"]["content"]
    return teacher_response

def ollama_judge(seed:str, text_chunk:str, history:str) -> int:
    content = ("Overall topic: " + text_chunk +
               "\n Seed question: " + seed +
               "\n Conversation History: " + history)
    client = ollama.Client(host="http://atlas1api.eurecom.fr:8019")
    response = client.chat(model="llama3.1", messages=[{"role": "system", "content": judge_role},
                                                       {"role": "user", "content": content}])
    judge_response = response["message"]["content"]
    return judge_response