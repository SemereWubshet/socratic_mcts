import pathlib

from transformers import TrainingArguments

if __name__ == "__main__":
    training_args = TrainingArguments(output_dir=pathlib.Path("/tmp"), num_train_epochs=1, learning_rate=1e-5,
                                      gradient_checkpointing=True)