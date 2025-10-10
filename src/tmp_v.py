import unsloth
import pathlib

from datasets import Dataset
from transformers import TrainingArguments, Trainer

from socratic_rl_peft import ActionValueFn

if __name__ == "__main__":
    action_vf_path = pathlib.Path("/home/gatti/socratic-rl/trial-0/train/iteration_0/action_value_fn/")

    dataset = Dataset.load_from_disk("/home/gatti/socratic-rl/trial-0/train/iteration_0/vf_training/it_5/dataset/")
    dataset = dataset.train_test_split(test_size=0.25).shuffle()
    print(len(dataset['test']))
    print(len(dataset['train']))

    training_args = TrainingArguments(
        num_train_epochs=1,
        output_dir=pathlib.Path("/tmp/trainer_output"),
        overwrite_output_dir=True,
        learning_rate=1e-5,
        gradient_checkpointing=True
    )
    action_value_fn = ActionValueFn(action_vf_path, max_length=1024, gpu="cuda")
    action_value_fn.load()
    trainer = Trainer(
        model=action_value_fn.model, args=training_args, train_dataset=dataset["train"], eval_dataset=dataset["test"]
    )
    results = trainer.train()
    print(results)

    output = trainer.predict(dataset["test"])
    print(output)


    # print(train)
    # print(len(test))


    # action_value_fn = ActionValueFn(str(action_vf_path), max_length=1024, gpu="cuda")
    #
    # v = action_value_fn([
    #     {"role": "user", "content": "what's the capital of Brazil?"},
    #     {"role": "assistant", "content": "The capital of Brazil is Brasilia you dumb student!"}
    # ])
    #
    # print(v)
    #
    # v = action_value_fn([
    #     {
    #         "role": "user",
    #         "content": "Does the amount of UV light absorbed by a molecule always result in a chemical reaction?",
    #     },
    #     {
    #         "role": "assistant",
    #         "content": "That's an interesting question. Can you think of any situations where a molecule might absorb "
    #                    "UV light without undergoing a chemical reaction?",
    #     },
    #     {
    #         "role": "user",
    #         "content": "Absorption of UV light by a molecule can result in relaxation pathways, where the energy is "
    #                    "dissipated as heat or emitted as light. This is different from photochemical reactions, where "
    #                    "the absorbed energy leads to bond-breaking events. So, it is possible that the absorbed UV "
    #                    "light does not always result in a chemical reaction.",
    #     },
    #     {
    #         "role": "assistant",
    #         "content": "That's a great start! Can you think of any specific examples where a molecule absorbs UV "
    #                    "light but does not undergo a chemical reaction? What might cause the energy to be dissipated "
    #                    "instead of used to break bonds?",
    #     },
    #     {
    #         "role": "user",
    #         "content": "Fluorescence and phosphorescence are examples where absorbed UV light is re-emitted as light "
    #                    "of longer wavelengths rather than causing a chemical reaction. The energy can also be "
    #                    "dissipated as heat through non-radiative processes. In these cases, the molecule returns to "
    #                    "its ground state without undergoing any chemical change. The efficiency of photochemical "
    #                    "reactions can be quantified by the quantum yield, which is the ratio of the number of "
    #                    "photochemical events to the number of photons absorbed.",
    #     },
    #     {
    #         "role": "assistant",
    #         "content": "It seems like you have a solid understanding of the topic. Would you like to explore how "
    #                    "the quantum yield can be measured or what factors influence it?",
    #     }
    # ])
    #
    # print(v)
