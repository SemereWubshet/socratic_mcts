from datasets import Dataset

if __name__ == "__main__":
    d = Dataset.load_from_disk("/home/gatti/socratic-rl/trial-0/train/iteration_0/vf_training/it_5/dataset")
    print(d["history"][0])
    # print(d["input_ids"][0])
    print(d["labels"][0])
