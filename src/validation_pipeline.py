import argparse
import json
import pathlib
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
    # Split text into chunks of size chunk_size
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

def pipeline(input_name:TextIO, output_name:TextIO, conversation_number) -> None:
    """Assemble tools to build a Socratic pedagogical dialogue"""
    contents = input_name.read()
    text_chunks = split_into_chunks(contents, chunk_size)

    exchanges = []
    results = []

    for index in range(conversation_number):
        text_chunk = text_chunks[index]
        exchange = generate_exchange(text_chunk)
        exchanges.append(exchange)

    exchanges_dump = [history.get_history() for history in exchanges]
    json.dump(exchanges_dump, args.o, indent=4)
    args.o.close()

def abc(inny:TextIO, ciby:TextIO, outy:TextIO):
    contents = ciby.read()
    print(type(contents))
    print(contents)
    return contents

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', required=True, help='', type=argparse.FileType('r'))
    parser.add_argument('-o', required=True, help='', type=argparse.FileType('w'))
    parser.add_argument('-c', required=True, help='', type=argparse.FileType('r'))


    # Depth of socratic conversation
    depth = 1

    # Chunk size of splits in input file
    chunk_size = 5000
    args = parser.parse_args()

    a = abc(args.i, args.c, args.o)

    # h = ChatHistory()
    # h.add_student("Why is the sky blue?")
    # h.add_teacher("Could it be the angle of the sun?")
    # h.add_student("Perhaps the blue light is spread by the atmosphere giving it a blue tint.")
    # h.add_teacher("Exactly! That's also the reason the sky turns orange during sunrise and sunset.")
    #
    # j = ChatHistory()
    # j.add_student("What is love?")
    # j.add_teacher("What do you think it is?")
    # j.add_student("Perhaps when we are willing to do anything for another person?")
    # j.add_teacher("Of course!.")
    #
    #
    # a = str(h)
    # b = h.get_history()
    # c = j.get_history()
    # d = [b,c]
    # json.dump(d, args.o, indent=4)
    # judge_response = judge("The sky", "The sky is blue", h)
    # print("I'm judge response: " + str(judge_response))
    # x = [judge_response]
    # for result in x:
    #     with open('datasets/' + 'results_val.txt', 'w') as f:
    #         f.write(str(result))

    # file_path = 'datasets/' + 'eval.json'
    # with open(file_path, 'r') as file:
    #     data = json.load(file)

    # print(type(data), type(data[0]))
    # print(data[0])
    # print(type(data[0][0]))
    # print(data[0][0])
    # histories_list = [ChatHistory.from_history(exchange) for exchange in data]
    # out = [h.get_history() for h in histories_list]
    # json.dump(out, args.o, indent=4)




    # print("a", a)
    # print("b", b)


    # pipeline(args.i, args.o, 2)