import argparse
import pathlib
import json
import ollama
from openai import OpenAI

import shutil
from typing import Dict, List, Tuple
# from fsspec.caching import caches


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from statistics import mean, stdev
from /src/agents import EvaluationDataset
i = 7
go = pathlib.Path("helper")
going = pathlib.Path(go).with_name(f"{i}.json")
a = {"role": "Student",
     "content": "What are some unique characteristics of the Gymnocalycium genus?",
     "end": False}

human_eval_dataset = EvaluationDataset.model_validate_json(human_eval_path.read_text())

going.write_text(json.dump(a,'w', indent=4))