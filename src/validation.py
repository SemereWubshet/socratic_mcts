import argparse
import json
import pathlib
import random
from typing import TextIO

import query_tools as qt
from src.conversation_generator import *

# Remove this function: I can just qt.ollama_gen_seed() directly
def gen_seed_topic(text_chunk:str) -> str:
    """Call to llm to generate a specific seed topic given a chunk"""
    seed_topic = qt.ollama_gen_seed(text_chunk)
    return seed_topic


# class ChatHistory:
#     """The history will store a sequence of student and teacher queries."""
#     def __init__(self):
#         self.history = []
#
#     # Add at the beginning
#     def add_text_chunk(self, query:str) -> None:
#         self.history.append({'role': 'text_chunk', 'query': query})
#     def add_student(self, query:str) -> None:
#         self.history.append({'role': 'student', 'query': query})
#     def add_teacher(self, query:str) -> None:
#         self.history.append({'role': 'teacher', 'query': query})
#
#     # Add at the end
#     def add_eval(self, query:int) -> None:
#         self.history.append({'role': 'evaluation', 'query': query})
#     def get_history_list(self) -> list:
#         return self.history
#     def __str__(self) -> str:
#         history_str = []
#         for entry in self.history[1:-1]:
#             role = entry['role']
#             query = entry['query']
#             history_str.append(f"{role.capitalize()}: {query}")
#         return "\n".join(history_str)
#     def is_empty(self) -> bool:
#         return len(self.history) == 0
#     def get_text_chunk(self) -> str:
#         return self.history[0]["query"]
#     def get_seed(self) -> str:
#         return self.history[1]["query"]
#     def get_eval(self) -> str:
#         if self.history[-1]['role'] == 'evaluation':
#             return self.history[-1]["query"]
#         else:
#             return ""
#
#     @classmethod
#     def from_history(cls, exchanges: list) -> 'ChatHistory':
#         """Regenerate a ChatHistory object from a string representation of exchanges."""
#         chat_history = cls()  # Create a new instance of ChatHistory
#
#         for entry in exchanges:
#             if entry['role'] == 'text_chunk':
#                 chat_history.add_text_chunk(entry['query'])
#             elif entry['role'] == 'student':
#                 chat_history.add_student(entry['query'])
#             elif entry['role'] == 'teacher':
#                 chat_history.add_teacher(entry['query'])
#             elif entry['role'] == 'evaluation':
#                 chat_history.add_eval(entry['query'])
#
#         return chat_history
#
# def teacher(history:ChatHistory) -> str:
#     """Generate teacher response based on history"""
#     teacher_response = qt.ollama_gen_teacher_response(str(history))
#     return teacher_response
#
# def student(seed:str, history:ChatHistory) -> str:
#     """Generate student response based on seed and history"""
#     if history.is_empty():
#         student_response = qt.ollama_gen_soc_question(seed)
#     else:
#         student_response = qt.ollama_gen_student_response(seed, str(history))
#
#     return student_response


def judge(history:ChatHistory) -> int:
    """Judge whether the teacher displayed correct Socratic behavior"""
    seed = history.get_seed()
    text_chunk = history.get_text_chunk()
    history_str = str(history)
    judge_response = qt.openai_gen_judge(seed, text_chunk, history_str) # Using open_ai at the moment
    return judge_response

def pipeline(input_name:TextIO, output_name:TextIO) -> None:
    all_conversations = json.load(input_name)
    # conversations_list = [ChatHistory.from_history(exchange) for exchange in all_conversations]
    conversations_list = []
    conversation_dump = []
    for conversation in all_conversations:
        history = ChatHistory.from_history(conversation)
        result = judge(history)
        history.add_eval(result)
        conversations_list.append(history)
        conversation_dump.append(history.get_history_list())

    json.dump(conversation_dump, output_name, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', required=True, help='', type=argparse.FileType('r'))
    parser.add_argument('-o', required=True, help='', type=argparse.FileType('w'))
    args = parser.parse_args()

    # Run pipeline
    pipeline(args.i, args.o)


    # with open('caches/conversations.json', 'r') as f:
    #     conversations_list = json.load(f)
    # #
    # hist_list = [ChatHistory.from_history(convo) for convo in conversations_list]
    # with open('saves/save.txt', 'w') as f:
    #
    #     for hist in hist_list:
    #         # f.write("New conversation\n")
    #         f.write(str(hist))
    #         f.write("\n=================\n")
    # out = [str(hist) for hist in hist_list]
    # print(out)

