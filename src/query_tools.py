import random
import ollama
import pathlib
from typing import Dict, List, Any
import json
from conversation_generator import ChatHistory # Remove if it fails due to circular imports
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

def gen_dataset(conversations: List[ChatHistory]) -> List[Dict[str, Any]]:
    client = ollama.Client(host="http://atlas1api.eurecom.fr:8019")

    system_prompt = pathlib.Path("./templates/judge.txt").read_text(encoding="UTF-8")
    answer_prompt = pathlib.Path("./templates/answer.txt").read_text(encoding="UTF-8")

    dataset = []

    for conversation in conversations:
        text_chunk = conversation.get_text_chunk()
        seed_question = conversation.get_seed()

        content = answer_prompt.format(context=text_chunk, question=seed_question)

        response = client.chat(model="mistral-nemo:12b-instruct-2407-fp16",
                               messages=[{"role": "user", "content": content}],
                               options={
                                   "num_ctx": 16_000,
                                   "temperature": 0.1,
                               })

        topics = response["message"]["content"]

        eval_query = f"# Main topics\n{topics}\n\n# Chat history\n{conversation}"
        response = client.chat(model="mistral-nemo:12b-instruct-2407-fp16",
                               messages=[
                                   {"role": "system", "content": system_prompt},
                                   {"role": "user", "content": eval_query}],
                               options={
                                   "num_ctx": 32_000,
                                   "temperature": 0.1,
                               })

        evaluation: str = response["message"]["content"]
        as_json = json.loads(evaluation)
        dataset.append({"topics": topics,
                        "history": conversation,
                        "reason": as_json["feedback"],
                        "assessment": as_json["assessment"]})

    return dataset