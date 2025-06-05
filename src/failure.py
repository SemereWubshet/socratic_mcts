import argparse
import glob
import pathlib
from collections import defaultdict

import pandas as pd
from bokeh.io import export_svg, curdoc
from bokeh.models import ColumnDataSource
from bokeh.palettes import Category10
from bokeh.plotting import figure
from bokeh.themes import Theme
from bokeh.transform import dodge
from transformers import pipeline

from schemas import EvaluationDataset

VISUAL_SETTINGS = {
    "eurecom-ds/phi-3-mini-4k-socratic": {
        "color": Category10[10][0],
        "display_name": "Socratic LLM"
    },
    "mistral-small3.1:24b": {
        "color": Category10[10][1],
        "display_name": "MistralSmall 3.1"
    },
    "llama3.3:70b": {
        "color": Category10[10][2],
        "display_name": "Llama3.3"
    },
    "gemma3:27b": {
        "color": Category10[10][3],
        "display_name": "Gemma 3"
    },
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("INPUT_DIR", type=pathlib.Path)
    parser.add_argument("OUTPUT_DIR", type=pathlib.Path)
    parser.add_argument("--bokeh-theme", type=pathlib.Path, default=pathlib.Path("./theme.yml"))
    args = parser.parse_args()

    curdoc().theme = Theme(filename=args.bokeh_theme)

    failures = defaultdict(list)

    for name in glob.glob(str(args.INPUT_DIR / "eval_*.json")):
        with open(name) as f:
            dataset = EvaluationDataset.model_validate_json(f.read())
            if dataset.metadata.max_interactions >= 16:
                for e in dataset.evaluations:
                    if not e.assessment:
                        summary = e.feedback.split("# Verdict Summary")[-1].strip()
                        failures[dataset.metadata.teacher_llm].append(summary)

    classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

    labels = [
        "All main topics addressed",
        "Used Socratic method",
        "Resolved the opening question",
        "Adapted to student responses",
        "Helped deepen student understanding"
    ]

    all_data = []
    for teacher_llm, summaries in failures.items():
        out = classifier(summaries, labels, multi_label=True)

        label_scores = defaultdict(float)
        for result in out:
            for label, score in zip(result["labels"], result["scores"]):
                label_scores[label] += score

        normalized = {label: score / len(out) for label, score in label_scores.items()}

        for l in labels:
            all_data.append(
                {
                    "label": l,
                    "llm": teacher_llm,
                    "score": normalized.get(l, 0.0)
                }
            )

    # Create DataFrame
    df = pd.DataFrame(all_data)

    unique_llms = df["llm"].unique()
    colors = Category10[max(3, len(unique_llms))]

    # Plot
    p = figure(
        x_range=labels,
        y_range=(0, 1.05),
        height=400,
        width=100 + 100 * len(unique_llms),
        output_backend="svg",
        x_axis_label="Pedagogical Properties",
        y_axis_label="Normalized Presence\nScore (from 0 to 1)"
    )

    for i, llm in enumerate(unique_llms):
        llm_df = df[df["llm"] == llm].set_index("label")
        source = ColumnDataSource(data={
            "label": labels,
            "score": [llm_df.loc[l, "score"] if l in llm_df.index else 0.0 for l in labels]
        })
        p.vbar(
            x=dodge("label", -0.3 + i * (0.6 / len(unique_llms)), range=p.x_range),
            top="score",
            width=0.6 / len(unique_llms),
            source=source,
            legend_label=llm,
            color=colors[i % len(colors)]
        )

    p.xgrid.grid_line_color = None
    p.xaxis.major_label_orientation = 0.9
    p.legend.orientation = "horizontal"
    p.legend.location = "top_center"

    export_svg(p, filename=str(args.OUTPUT_DIR / "pedagogical_properties.svg"))
