import random
import ollama
import pathlib
import os
import google.generativeai as genai
from openai import OpenAI

# with open('templates/query_generator.txt', 'r') as f: # Make a keypoints.txt role
#     keypoints_role = f.read()

keypoints_role = "Given the piece of text below, extract major keypoints necessary for comprehensive understanding."

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

with open('templates/answer.txt', 'r') as f:
    answer_role = f.read()

"""Interaction types"""

INTERACTION_TYPES = (
    "Demand deeper clarification about one of the major points on the topic.",
    "Request further explanations that go beyond the original text.",
    "Make misleading claims due to misunderstanding on one or more of the topics.",
    "Act confused about one of the major points, thus requiring further explanation from the teacher.",
    "Demonstrate inability to connect major points.",
    "Suggest a different understanding of a major point so to lead to a discussion about its validity.",
    "Request examples or applications of a major point in practical, real-world scenarios.",
    "Request the comparison to major points with similar or contrasting concepts.",
    "Pose \"what-if\" questions to explore the implications of the major point in various contexts.",
    "Question the foundational assumptions of the major point, prompting justification or re-explanation.",
    "Request an explanation of the major point in simpler terms or using analogies.",
    "Demonstrate understanding of some basic concepts but struggle to connect them to the broader major point.",
    "Ask questions that are tangentially related to the major point, requiring the teacher to refocus the conversation "
    "while addressing the inquiry.",
    "Ask for a detailed breakdown of a process or concept related to the major point.",
    "Ask if there are any arguments or evidence against the major point, prompting critical evaluation.",
    "Make overly broad statements about the major point, requiring clarification or correction.",
)

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

def ollama_gen_key_points(content):
    client = ollama.Client(host="http://atlas1api.eurecom.fr:8019")
    response = client.chat(model="llama3.1", messages=[{"role": "system", "content": keypoints_role},
                                                       {"role": "user", "content": content}])
    keypoints = response["message"]["content"]
    return keypoints

def ollama_gen_soc_question(content):
    # keypoints = ollama_gen_key_points(content)
    # client = ollama.Client(host="http://atlas1api.eurecom.fr:8019")
    # response = client.chat(model="llama3.1", messages=[{"role": "system", "content": query_role},
    #                                                    {"role": "user", "content": keypoints}])

    base_prompt = pathlib.Path("./templates/seed.txt").read_text(encoding="UTF-8")
    interaction_type = random.choice(INTERACTION_TYPES)
    content = base_prompt.format(context=content, interaction_type=interaction_type)

    client = ollama.Client(host="http://atlas1api.eurecom.fr:8019")
    response = client.chat(model="mistral-nemo:12b-instruct-2407-fp16",
                           messages=[{"role": "user", "content": content}])

    question = response["message"]["content"]
    return question

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
    response = client.chat(model="llama3.1", messages=[{"role": "system", "content": teacher_role},
                                                       {"role": "user", "content": content}])
    teacher_response = response["message"]["content"]
    return teacher_response

def ollama_answer(text_chunk:str, seed:str):
    content = answer_role.format(context=text_chunk, question=seed)
    client = ollama.Client(host="http://atlas1api.eurecom.fr:8019")
    response = client.chat(model="mistral-nemo:12b-instruct-2407-fp16", messages=[{"role": "user", "content": content}])
    answers = response["message"]["content"]
    return answers

def ollama_judge(seed:str, text_chunk:str, history:str) -> int:
    true_answer = ollama_answer(text_chunk, seed)
    content = ("Overall topic: " + text_chunk +
               "\n Seed question: " + seed +
               "\n Main Topic: " + true_answer +
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
