import argparse
import json
import pathlib
import copy
from typing import TextIO

import query_tools as qt


def gen_seed_topic(text_chunk:str) -> str:
    """Call to llm to generate a specific seed topic given a chunk"""
    seed_topic = qt.ollama_gen_seed(text_chunk)
    return seed_topic


class ChatHistory:
    """The history will store a sequence of student and teacher queries."""
    def __init__(self):
        self.history = []
    def add_student(self, query:str) -> None:
        self.history.append({'role': 'student', 'query': query})
    def add_teacher(self, query:str) -> None:
        self.history.append({'role': 'teacher', 'query': query})
    def get_history(self) -> list:
        return self.history
    # def __str__(self) -> str:
    #     return super().__str__()
    def __str__(self) -> str:
        history_str = []
        for entry in self.history:
            role = entry['role']
            query = entry['query']
            history_str.append(f"{role.capitalize()}: {query}")
        return "\n".join(history_str)
    def is_empty(self) -> bool:
        return len(self.history) == 0
    def get_first(self) -> dict:
        return self.history[0] if self.history else None
    def get_last(self) -> dict:
        return self.history[-1] if self.history else None

def teacher(history:ChatHistory) -> str:
    """Generate teacher response based on history"""
    teacher_response = qt.ollama_gen_teacher_response(str(history))
    return teacher_response

def student(seed:str, history:ChatHistory) -> str:
    """Generate student response based on seed and history"""
    if history.is_empty():
        student_response = qt.ollama_gen_soc_question(seed)
    else:
        student_response = qt.ollama_gen_student_response(seed, str(history))

    return student_response


def judge(seed:str, text_chunk:str, history:ChatHistory) -> bool:
    """Judge whether the teacher displayed correct Socratic behavior"""
    judge_response = qt.ollama_judge(seed, text_chunk, str(history))
    return judge_response

def generate_exchanges(seed, history:ChatHistory, iter:int, depth:int) -> list:
    """Generate iter Socratic responses by the teacher"""
    if depth == 0: return [0];
    student_query = student(seed, history)
    history.add_student(student_query) # Student response

    histories = []

    for _ in range(iter): # Multiple teacher responses
        new_history = copy.deepcopy(history)
        teacher_query = teacher(new_history)

        new_history.add_teacher(teacher_query)
        item_histories = generate_exchanges(seed, new_history, iter, depth - 1)
        histories.append(new_history)

    for item in histories:
        item_histories = generate_exchanges(seed, item, iter, depth - 1)

    return histories

# def generate_tree(tree_depth:int, tree_width:int, text_chunk:str) -> list:
#     seed = gen_seed_topic(text_chunk)
#     history = ChatHistory()
#
#     history_list = generate_exchanges(seed, history, tree_width)
#
#
#     for _ in range(tree_depth - 1):
#         list = generate_exchanges(seed, history, tree_width)
#
#     return list

def split_into_chunks(text, chunk_size):
    """Split a given text file into manageable pieces"""
    # Split text into chunks of size chunk_size
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

def pipeline(input_name:TextIO, output_name:TextIO) -> None:
    """Assemble tools to build a Socratic pedagogical dialogue"""
    contents = input_name.read()
    text_chunks = split_into_chunks(contents, chunk_size)

    for text_chunk in text_chunks:
        exchanges = generate_exchanges()

    # Save the results somehow as well!

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', required=True, help='', type=argparse.FileType('r'))
    parser.add_argument('-o', required=True, help='', type=argparse.FileType('w'))

    # Depth of socratic conversation
    depth = 2

    # Chunk size of splits in input file
    chunk_size = 50000
    args = parser.parse_args()
    pipeline(args.i, args.o)