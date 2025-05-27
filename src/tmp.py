from socratic_rl_peft import Gemma

if __name__ == "__main__":
    s = Gemma()

    chat = [
        {"role": "user", "content": "What is the answer to life the universe and everything?"},
        {"role": "model", "content": "Arrr, 'tis 42,"},
    ]
    s.query(chat)
    pass
