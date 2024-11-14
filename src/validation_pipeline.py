import argparse
import json
import pathlib


def gen_seed_topic(text_chunk:str) -> str:
    """Call to llm to generate a specific seed topic given a chunk"""
    return ""


class ChatHistory:
    def add_student(self, query:str) -> None:
        ...
    def add_teacher(self, query:str) -> None:
        ...

    def __str__(self) -> str:
        return super().__str__()


def teacher(history:ChatHistory) -> str:
    ...

def student(seed:str, history:ChatHistory) -> str:
    ...

def judge(seed:str, text_chunk:str, history:ChatHistory) -> bool:

    ...


def generate_exchange(text_chunk:str) -> (ChatHistory, bool):
    text_chunk = ''
    seed = gen_seed_topic(text_chunk)
    history = ChatHistory()

    for _ in range(5):
        student_query = student(seed, history)
        history.add_student(student_query)

        teacher_query = teacher(history)
        history.add_teacher(teacher_query)

    result = judge(seed, text_chunk, history)
    return history, result

def split_into_chunks(text, chunk_size):
    # Split text into chunks of size chunk_size
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

def pipeline(output_name:pathlib.Path, input_name:pathlib.Path) -> None:
    with open(input_name, 'r') as f:
        contents = f.read()
    text_chunks = split_into_chunks(contents, 1000)
    exchanges = []
    for text_chunk in text_chunks:
        exchange = generate_exchange(text_chunk)
        exchanges.append(exchange)

    with open(output_name, 'w') as f:
        json.dump(exchanges, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', required=True, help='', type=argparse.FileType('r'))
    parser.add_argument('-o', required=True, help='', type=argparse.FileType('w'))

    args = parser.parse_args()

    pipeline(args.o, args.i)
