import argparse
import itertools
import json
import random
from typing import TextIO

from datasets import load_dataset

import src.query_tools as qt
import json
import pathlib
from typing import Dict, List, Any
import ollama
import datasets


# Remove this function: I can just qt.ollama_gen_seed() directly
def gen_seed_question(text_chunk:str) -> str:
    """Call to llm to generate a specific seed topic given a chunk"""
    seed_question = qt.ollama_gen_soc_question(text_chunk)
    return seed_question


class ChatHistory:
    """The history will store a sequence of student and teacher queries."""
    def __init__(self):
        self.history = []

    # Add at the beginning
    def add_text_chunk(self, query:str) -> None:
        self.history.append({'role': 'text_chunk', 'query': query})
    def add_student_type(self, student_type:int) -> None:
        self.history.append({'role': 'student_type', 'query': student_type})
    def add_student(self, query:str) -> None:
        self.history.append({'role': 'student', 'query': query})
    def add_teacher(self, query:str) -> None:
        self.history.append({'role': 'teacher', 'query': query})

    # Add at the end
    def add_eval(self, query:int) -> None:
        self.history.append({'role': 'evaluation', 'query': query})
    def get_history_list(self) -> list:
        return self.history
    def __str__(self) -> str:
        history_str = []
        for entry in self.history:
            if entry['role'] == "student" or entry['role'] == "teacher":
                history_str.append(f"{entry['role'].capitalize()}: {entry['query']}")
        # for entry in self.history[1:-1]:
        #     role = entry['role']
        #     query = entry['query']
        #     history_str.append(f"{role.capitalize()}: {query}")
        # if self.history[-1]['role'] != 'evaluation' and self.history[-1]['role'] != 'text_chunk':
        #     role = self.history[-1]['role']
        #     query = self.history[-1]['query']
        #     history_str.append(f"{role.capitalize()}: {query}")
        return "\n".join(history_str)
    def is_empty(self) -> bool:
        return len(self.history) == 0
    def get_text_chunk(self) -> str:
        return self.history[0]["query"]
    def get_student_type(self) -> int:
        return self.history[1]["query"]
    def get_seed(self) -> str:
        return self.history[2]["query"]
    def get_eval(self) -> str:
        if self.history[-1]['role'] == 'evaluation':
            return self.history[-1]["query"]
        else:
            return ""

    @classmethod
    def from_history(cls, exchanges: list) -> 'ChatHistory':
        """Regenerate a ChatHistory object from a string representation of exchanges."""
        chat_history = cls()  # Create a new instance of ChatHistory

        for entry in exchanges:
            if entry['role'] == 'text_chunk':
                chat_history.add_text_chunk(entry['query'])
            elif entry['role'] == 'student_type':
                chat_history.add_student_type(entry['query'])
            elif entry['role'] == 'student':
                chat_history.add_student(entry['query'])
            elif entry['role'] == 'teacher':
                chat_history.add_teacher(entry['query'])
            elif entry['role'] == 'evaluation':
                chat_history.add_eval(entry['query'])

        return chat_history

def teacher(history:ChatHistory) -> str:
    """Generate teacher response based on history"""
    teacher_response = qt.ollama_gen_teacher_response(str(history))
    return teacher_response


def student(text_chunk:str, seed_question:str, history:ChatHistory) -> str:
    """Generate student response based on seed and history"""
    student_response = qt.ollama_gen_student_response(text_chunk, seed_question, str(history), history.get_student_type())
    return student_response


# def judge(seed_topic:str, text_chunk:str, history:ChatHistory) -> int:
#     """Judge whether the teacher displayed correct Socratic behavior"""
#     judge_response = qt.ollama_judge(seed_topic, text_chunk, str(history))
#     return judge_response

def generate_exchange(text_chunk:str, depth: int = 2) -> ChatHistory:
    """Generate Socratic dialogue between a student and a teacher"""
    seed_question = gen_seed_question(text_chunk)
    history = ChatHistory()
    history.add_text_chunk(text_chunk)
    history.add_student_type(7)
    history.add_student(seed_question)

    for _ in range(depth - 1):
        teacher_query = teacher(history)
        history.add_teacher(teacher_query)

        student_query = student(text_chunk, seed_question, history)
        history.add_student(student_query)

    teacher_query = teacher(history)
    history.add_teacher(teacher_query)

    return history

def split_into_chunks(text, chunk_size=2000):
    """Split a given text file into manageable pieces"""
    # chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
    topics = text.split("Title: ")  # To avoid having chunks with two distinct subjects
    chunks=[]
    for topic in topics:
        for i in range(0, len(topic), chunk_size):
            chunks.append(topic[i:i + chunk_size])
    random.shuffle(chunks) # Randomize chunks
    return chunks

def pipeline(input_name:TextIO, output_name:TextIO, number_of_conversations, depth: int = 2) -> None:
    """Assemble tools to build a Socratic pedagogical dialogue"""
    # contents = input_name.read()
    contents = load_dataset("wikimedia/wikipedia", "20231101.simple")['train']

    exchanges = []

    for index in range(number_of_conversations):
        page = random.choice(contents)
        page_text = page["text"]
        text_chunks = split_into_chunks(page_text)
        text_chunk = random.choice(text_chunks)
        exchange = generate_exchange(text_chunk, depth)
        exchanges.append(exchange)

    rated_exchanges = qt.openai_gen_dataset(exchanges)
    output_data = []
    for exchange in rated_exchanges:
        output_data.append({"history": exchange["history"].get_history_list(),
                            "topics":exchange["topics"], "reason": exchange["reason"],
                            "assessment": exchange["assessment"]})
    output_name.write(json.dumps(output_data, indent=4))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', required=True, help='Input textual data on covered topics', type=argparse.FileType('r'))
    parser.add_argument('-o', required=True, help='Output json dataset with student-teacher interactions', type=argparse.FileType('w'))
    parser.add_argument('-num', required=False, help='Number of conversations to generate', type=int)
    args = parser.parse_args()

    # Attributes of socratic conversations
    depth = 2 # Depth of conversations
    # chunk_size = 1000 # Chunk size of splits in input file
    num_conversations = 2 # Number of conversations
    llm = 'openai' # LLM used by all functions
    if args.num:
        num_conversations = args.num

    # Run pipeline
    pipeline(args.i, args.o, num_conversations, depth)