from socratic_rl_peft import Phi4

if __name__ == "__main__":
    s = Phi4(adapter_path="/home/gatti/test-2/train/iteration_0/policy_fn/", device="cuda:0")

    chat = [
        {"role": "user", "content": "What is the answer to life the universe and everything?"},
        {"role": "model", "content": "Arrr, 'tis 42,"},
        {"role": "user", "content": "Where does this comes from?"}
    ]
    a = s.query(chat)
    print(a)
    pass
