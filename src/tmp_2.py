import argparse
import pathlib

from vf_dataset_build_2 import ActionValueFn

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("V_PATH", type=pathlib.Path)
    args = parser.parse_args()

    action_value_fn = ActionValueFn(str(args.V_PATH))

    print("Q: What's the capital of Brazil?")

    result = action_value_fn([
        {"role": "user", "content": "What's the capital of Brazil?"},
        {"role": "assistant", "content": "The capital of Brazil is Brasilia."}
    ])

    print()
    print("A: The capital of Brazil is Brasilia.")
    print(f"q(s,a) = {float(result)}")

    result = action_value_fn([
        {"role": "user", "content": "What's the capital of Brazil?"},
        {"role": "assistant", "content": "Can you think about planned cities built in Brazil in 1950s?"}
    ])
    print()
    print("Can you think about planned cities built in Brazil in 1950s?")
    print(f"q(s,a) = {float(result)}")

    result = action_value_fn([
        {"role": "user", "content": "What's the capital of Brazil?"},
        {"role": "assistant", "content": "Brazil moved its capital from Rio in 1950s. Can you think of the reasons "
                                         "for the government to want do so?"}
    ])
    print()
    print("Brazil moved its capital from Rio in 1950s. Can you think of the reasons for the government to want do so?")
    print(f"q(s,a) = {float(result)}")

    result = action_value_fn([
        {"role": "user", "content": "What's the capital of Brazil?"},
        {"role": "assistant", "content": "The capital of Brazil was Rio de Janeiro until 1950s. On that time, "
                                         "Brazil was mostly a coastal country despite its vast territory. "
                                         "The Brazilian government wanted to move the capital inwards the country "
                                         "so to make it more strategically positioned. Can you think about planned "
                                         "Brazilian cities built during that period?"}
    ])
    print()
    print("The capital of Brazil was Rio de Janeiro until 1950s. On that time, Brazil was mostly a coastal country "
          "despite its vast territory. The Brazilian government wanted to move the capital inwards the country so to "
          "make it more strategically positioned. Can you think about planned Brazilian cities built during that "
          "period?")
    print(f"q(s,a) = {float(result)}")
