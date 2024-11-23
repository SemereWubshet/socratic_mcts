import argparse
import json
import pathlib
import random
from typing import TextIO

import query_tools as qt


# Remove this function: I can just qt.ollama_gen_seed() directly
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

    @classmethod
    def from_history(cls, exchanges: list) -> 'ChatHistory':
        """Regenerate a ChatHistory object from a string representation of exchanges."""
        # history_data = json.loads(exchanges)  # Parse the string into a list of dictionaries
        chat_history = cls()  # Create a new instance of ChatHistory

        for entry in exchanges:
            if entry['role'] == 'student':
                chat_history.add_student(entry['query'])
            elif entry['role'] == 'teacher':
                chat_history.add_teacher(entry['query'])

        return chat_history

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


def judge(seed:str, text_chunk:str, history:ChatHistory) -> int:
    """Judge whether the teacher displayed correct Socratic behavior"""
    judge_response = qt.ollama_judge(seed, text_chunk, str(history))
    return judge_response

def generate_exchange(text_chunk:str) -> ChatHistory:
    """Generate Socratic dialogue between a student and a teacher"""
    seed = gen_seed_topic(text_chunk)
    history = ChatHistory()

    for _ in range(depth):
        student_query = student(seed, history)
        history.add_student(student_query)

        teacher_query = teacher(history)
        history.add_teacher(teacher_query)

    return history

def split_into_chunks(text, chunk_size):
    """Split a given text file into manageable pieces"""
    chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
    random.shuffle(chunks) # Randomize chunks
    return chunks

def pipeline(input_name:TextIO, output_name:TextIO, number_of_conversations) -> None:
    conversations_list = json.load(input_name)
    histories_list = [ChatHistory.from_history(exchange) for exchange in conversations_list]

    results = [judge()]



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', required=True, help='', type=argparse.FileType('r'))
    parser.add_argument('-o', required=True, help='', type=argparse.FileType('w'))

    # Attributes of socratic conversations
    depth = 1 # Depth of conversations
    chunk_size = 5000 # Chunk size of splits in input file
    num_conversations = 5 # Number of conversations
    args = parser.parse_args()

    # Run pipeline
    pipeline(args.i, args.o, num_conversations)