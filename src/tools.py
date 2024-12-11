import argparse
from typing import List

from ollama import Client
from openai import OpenAI

from agents import LLM, OpenAIAgent, OllamaAgent


class LLMAction(argparse.Action):

    def __call__(
            self,
            parser: argparse.ArgumentParser,
            namespace: argparse.Namespace,
            values: List[str],
            option_string: List[str] = None
    ) -> None:
        if len(values) != 3:
            raise ValueError(f"Expected the model provider, the provider's information access, "
                             f"and the model name to use, but found {values}")

        provider, access_info, model = values

        if provider not in ("openai", "ollama"):
            raise ValueError(f"Only \"openai\" or \"ollama\" are the accepted providers. Found {provider}")

        client_llm: LLM
        if provider == "ollama":
            client = Client(host=access_info)
            client_llm = OllamaAgent(client=client, model=model)
        else:
            client = OpenAI(api_key=access_info)
            client_llm = OpenAIAgent(client=client, model=model)

        client_llm.healthcheck()

        setattr(namespace, self.dest, client_llm)
