import argparse

from socratic_rl_peft import Phi4

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--id", type=int, required=True)
    args = parser.parse_args()

    s = Phi4(adapter_path=f"/home/gatti/test-{args.id}/train/iteration_0/policy_fn/", device="cuda:0")

    chat = [
        {"role": "user", "content": "What is the answer to life the universe and everything?"},
        {"role": "model", "content": "Arrr, 'tis 42,"},
        {"role": "user", "content": "Where does this comes from?"}
    ]
    a = s.query(chat)
    print(a)
    pass
