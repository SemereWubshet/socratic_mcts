from socratic_rl_peft import Gemma

if __name__ == "__main__":
    s = Gemma(device="cuda:0")

    chat = [
        {"role": "user", "content": "What is the answer to life the universe and everything?"},
        {"role": "model", "content": "Arrr, 'tis 42,"},
        {"role": "user", "content": "Where does this comes from?"}
    ]
    a = s.query(chat)
    print(a)
    pass
