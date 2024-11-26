import ollama
import os
import google.generativeai as genai
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

def openai_gen_student_response(text_chunk, seed_question, history_str):
    client = OpenAI()
    content = ("Overall topic: " + text_chunk +
               "\n Seed question: " + seed_question +
               "\n Conversation History: " + history_str)
    # content = "Topic: " + seed + "\n Conversation History: " + history # if history else None
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

def ollama_gen_student_response(text_chunk, seed_question, history_str):
    content = ("Overall topic: " + text_chunk +
               "\n Seed question: " + seed_question +
               "\n Conversation History: " + history_str)
    # content = "Topic: " + seed + "\n Conversation History: " + history # if history else None
    client = ollama.Client(host="http://atlas1api.eurecom.fr:8019")
    response = client.chat(model="llama3.1", messages=[{"role": "system", "content": student_role},
                                                       {"role": "user", "content": content}])
    student_response = response["message"]["content"]
    return student_response

def ollama_gen_teacher_response(content):
    client = ollama.Client(host="http://atlas1api.eurecom.fr:8019")
    # print("\nTeacher role \n", teacher_role)
    # print("\nContent \n", content)
    response = client.chat(model="llama3.1", messages=[{"role": "system", "content": teacher_role},
                                                       {"role": "user", "content": content}])
    teacher_response = response["message"]["content"]
    # print("\nTeacher response \n", teacher_response)
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

"""Query functions by Gemini"""
api_key = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS')
genai.configure(api_key=api_key)
model_name = "learnlm-1.5-pro-experimental"
model = genai.GenerativeModel(model_name)

def prompter(role:str, content:str) -> str:
    message = (f"You shall play the role given below. Role: \n {role} \n"
               f"The person you are speaking to gives you the following content."
               f"Content: \n {content} \n")
    return message

def gemini_gen_soc_question(content):
    full_prompt = prompter(query_role, content)
    response = model.generate_content(full_prompt)
    questions = response.text
    return questions


def gemini_gen_seed(content):
    full_prompt = prompter(seed_role, content)
    response = model.generate_content(full_prompt)
    seed = response.text
    return seed

def gemini_gen_student_response(text_chunk, seed_question, history_str):
    content = ("Overall topic: " + text_chunk +
               "\n Seed question: " + seed_question +
               "\n Conversation History: " + history_str)
    full_prompt = prompter(student_role, content)
    response = model.generate_content(full_prompt)
    student_response = response.text
    return student_response

def gemini_gen_teacher_response(content):
    full_prompt = prompter(teacher_role, content)
    response = model.generate_content(full_prompt)
    teacher_response = response.text
    return teacher_response

def gemini_judge(seed:str, text_chunk:str, history:str) -> int:
    content = ("Overall topic: " + text_chunk +
               "\n Seed question: " + seed +
               "\n Conversation History: " + history)
    full_prompt = prompter(judge_role, content)
    response = model.generate_content(full_prompt)
    judge_response = response.text
    return judge_response

# print(gemini_gen_student_response("The sky is blue because blue light is refracted in the atmosphere",
#                                   "Why is the sky blue?",
#                                   "Student: Why is the sky blue"
#                                   "Teacher: What do you observe about the sunlight before it reaches the Earth's atmosphere?"
#                                   "Student: Sunlight is white (all colors combined)."
#                                   "Teacher: And what about when it is inside air or water?"))
