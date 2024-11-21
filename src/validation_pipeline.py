import argparse
import json
import pathlib
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


def judge(seed:str, text_chunk:str, history:ChatHistory) -> int:
    """Judge whether the teacher displayed correct Socratic behavior"""
    judge_response = qt.ollama_judge(seed, text_chunk, str(history))
    return judge_response

def generate_exchange(text_chunk:str) -> (ChatHistory, int):
    """Generate Socratic dialogue between a student and a teacher"""
    seed = gen_seed_topic(text_chunk)
    history = ChatHistory()

    for _ in range(depth):
        student_query = student(seed, history)
        history.add_student(student_query)

        teacher_query = teacher(history)
        history.add_teacher(teacher_query)

    result = judge(seed, text_chunk, history)
    print("Result inside the generate exchange: " + str(result))
    return history, result

def split_into_chunks(text, chunk_size):
    """Split a given text file into manageable pieces"""
    # Split text into chunks of size chunk_size
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

def pipeline(input_name:TextIO, output_name:TextIO) -> None:
    """Assemble tools to build a Socratic pedagogical dialogue"""
    contents = input_name.read()
    text_chunks = split_into_chunks(contents, chunk_size)

    exchanges = []
    results = []
    for text_chunk in text_chunks:
        exchange, result = generate_exchange(text_chunk)
        # print("I'm result: " + str(result))
        exchanges.append(exchange)
        results.append(result)

    exchanges_dump = [history.get_history() for history in exchanges]
    json.dump(exchanges_dump, args.o, indent=4)
    args.o.close()

    print("I'm many results: " + str(results))
    for result in results:
        with open('datasets/' + 'results_val.txt', 'w') as f:
            f.write("\n ======= \n" + result)

    # Save the results somehow as well!

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', required=True, help='', type=argparse.FileType('r'))
    parser.add_argument('-o', required=True, help='', type=argparse.FileType('w'))

    # Depth of socratic conversation
    depth = 1

    # Chunk size of splits in input file
    chunk_size = 1000
    args = parser.parse_args()


    # h = ChatHistory()
    # h.add_student("Why is the sky blue?")
    # h.add_teacher("Could it be the angle of the sun?")
    # h.add_student("Perhaps the blue light is spread by the atmosphere giving it a blue tint.")
    # h.add_teacher("Exactly! That's also the reason the sky turns orange during sunrise and sunset.")
    #
    # a = str(h)
    # b = h.get_history()
    # json.dump(str(h), args.o, indent=4)
    # judge_response = judge("The sky", "The sky is blue", h)
    # print("I'm judge response: " + str(judge_response))
    # x = [judge_response]
    # for result in x:
    #     with open('datasets/' + 'results_val.txt', 'w') as f:
    #         f.write("\n ===== " + str(result))


    # print("a", a)
    # print("b", b)


    pipeline(args.i, args.o)