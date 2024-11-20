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

def generate_exchanges(seed, root:ChatHistory, tree_width:int, tree_depth:int) -> list:
    """Generate iter Socratic responses by the teacher"""
    if tree_depth == 0:
        return []

    student_query = student(seed, root)
    root.add_student(student_query) # Student response

    exchanges = []
    for _ in range(tree_width): # Multiple teacher responses
        new_history = copy.deepcopy(root)

        teacher_query = teacher(new_history)
        new_history.add_teacher(teacher_query)
        histories_list.append(new_history)

        item_histories = generate_exchanges(seed, new_history, tree_width, tree_depth - 1)
        # exchanges.append({'history': new_history, 'children': item_histories}) # Nested dictionary

    return exchanges

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

    histories_list = []

    history = ChatHistory()
    exchanges = generate_exchanges("Why is the sky blue", history, 2, 2)

    exchanges_dump = [history.get_history() for history in histories_list]
    print(exchanges_dump)
    json.dump(exchanges_dump, args.o, indent=4)

    # pipeline(args.i, args.o)