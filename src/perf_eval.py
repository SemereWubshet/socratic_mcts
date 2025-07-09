import argparse
import glob
import pathlib

import numpy as np
import pandas as pd
import statsmodels.stats.proportion as smp
from bokeh.io import export_svgs, curdoc
from bokeh.models import ColumnDataSource, Whisker, Span
from bokeh.palettes import Category10
from bokeh.plotting import figure
from bokeh.themes import Theme
from bokeh.transform import dodge

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
    "gpt-4o": {
        "color": Category10[10][4],
        "display_name": "GPT-4o"
    },
    "models/learnlm-2.0-flash-experimental": {
        "color": Category10[10][5],
        "display_name": "LearnLM 2.0"
    },
}


def parse_model_name(model_name: str) -> float:
    import re
    match = re.search(r'(\d+(?:\.\d+)?)([bm])', model_name.lower())
    if not match:
        if model_name == "eurecom-ds/phi-3-mini-4k-socratic":
            return 3.82
        return np.nan
    size = float(match.group(1))
    unit = match.group(2)
    return size * (1 if unit == 'b' else 1e-3)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("INPUT_DIR", type=pathlib.Path)
    parser.add_argument("FIGS_DIR", type=pathlib.Path)
    parser.add_argument("--bokeh-theme", type=pathlib.Path, default=pathlib.Path("./theme.yml"))

    args = parser.parse_args()

    curdoc().theme = Theme(filename=args.bokeh_theme)
    input_dir = args.INPUT_DIR

    df = pd.DataFrame()
    for name in glob.glob(str(input_dir / "eval_*.json")):
        with open(name) as f:
            dataset = EvaluationDataset.model_validate_json(f.read())
            df_ = pd.DataFrame(
                {
                    "teacher_llm": [dataset.metadata.teacher_llm, ] * len(dataset),
                    "teacher_llm_size": [parse_model_name(dataset.metadata.teacher_llm), ] * len(dataset),
                    "max_interactions": [dataset.metadata.max_interactions, ] * len(dataset),
                    "evaluation_id": [e.id for e in dataset.evaluations],
                    "opening_question": [e.interaction.seed.question for e in dataset.evaluations],
                    "interaction_type": [e.interaction.seed.interaction_type for e in dataset.evaluations],
                    "student_type": [e.interaction.student_type for e in dataset.evaluations],
                    "conversation_length": [len(e.interaction.chat_history.root) for e in dataset.evaluations],
                    "assessment": [e.assessment for e in dataset.evaluations],
                }
            )
            df = pd.concat([df, df_], ignore_index=True)

    # ----- Success rate per model (table) ------
    filtered_df = df[df['max_interactions'] == df['max_interactions'].max()]

    summary_data = []
    for model_name, group in filtered_df.groupby('teacher_llm'):
        total = len(group)
        successes = group['assessment'].sum()
        success_rate = successes / total
        model_size = float(group['teacher_llm_size'].unique())

        ci_low, ci_upp = smp.proportion_confint(successes, total, alpha=0.05, method='wilson')

        summary_data.append({
            'LLM': VISUAL_SETTINGS[model_name]["display_name"],
            'Model Size (billion parameters)': model_size,
            'Success Rate': f"{success_rate:.1%}",
            'CI (95%)': f"[{ci_low:.1%} {ci_upp:.1%}]"
        })

    summary_df = pd.DataFrame(summary_data)
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(summary_df)

    # ---- Success Rate vs. Interaction Length ----
    p1 = figure(
        x_axis_label="Interaction Length",
        y_axis_label="Success Rate",
        height=250,
        width=400,
        output_backend="svg"
    )

    for teacher_llm, group in df.groupby("teacher_llm"):
        data = {
            "interaction_length": [],
            "success_rate": [],
            "ci_low": [],
            "ci_high": []
        }
        for n, g in group.groupby("max_interactions"):
            total = len(g)
            successes = g["assessment"].sum()
            success_rate = successes / total
            ci_low, ci_upp = smp.proportion_confint(successes, total, alpha=0.05, method='wilson')

            data["interaction_length"].append(n)
            data["success_rate"].append(success_rate)
            data["ci_low"].append(ci_low)
            data["ci_high"].append(ci_upp)

        source = ColumnDataSource(data)

        v = VISUAL_SETTINGS[teacher_llm]
        p1.varea(x="interaction_length", y1="ci_low", y2="ci_high", color=v["color"], source=source, alpha=0.3)
        p1.line(
            x="interaction_length",
            y="success_rate",
            source=source,
            line_width=3,
            color=v["color"],
            legend_label=v["display_name"]
        )

    export_svgs(p1, filename="./figs/success_rate_by_interaction_length.svg")

    # ---- Success Rate vs. Persona (student_type) ----
    pivot_df = (
        filtered_df.groupby(["student_type", "teacher_llm"])["assessment"]
        .mean()
        .unstack(fill_value=np.nan)  # Fill missing combinations with NaN
        .reset_index()
    )

    student_types = pivot_df["student_type"].unique()
    teacher_llms = [c for c in pivot_df.columns if c != "student_type"]
    persona_map = {s: f"SP{i}" for i, s in enumerate(student_types, start=1)}

    pivot_df["student_type_id"] = pivot_df["student_type"].map(persona_map)
    data = pivot_df.to_dict(orient="list")

    source = ColumnDataSource(data=data)

    p2 = figure(
        x_axis_label="Student persona",
        y_axis_label="Success rate",
        height=250,
        width=400,
        output_backend="svg",
        x_range=data["student_type_id"],
        y_range=(0, 1)
    )

    bar_width = 0.8 / max(1, len(teacher_llms))

    for i, teacher_llm in enumerate(teacher_llms):
        offset = -bar_width * len(teacher_llms) / 2 + bar_width / 2 + i * bar_width
        p2.vbar(x=dodge("student_type_id", offset, range=p2.x_range), top=teacher_llm, source=source,
                width=bar_width, color=VISUAL_SETTINGS[teacher_llm]["color"])

    p2.x_range.range_padding = 0.1
    p2.xgrid.grid_line_color = None

    print()
    print("Persona mapping")
    print(persona_map)

    export_svgs(p2, filename="./figs/success_rate_by_persona.svg")

    # ----- Success rate vs. opening interaction type -----
    pivot_df = (
        filtered_df.groupby(["interaction_type", "teacher_llm"])["assessment"]
        .mean()
        .unstack(fill_value=np.nan)  # Fill missing combinations with NaN
        .reset_index()
    )

    interaction_types = pivot_df["interaction_type"].unique()
    teacher_llms = [c for c in pivot_df.columns if c != "interaction_type"]
    interaction_map = {s: f"OQ{i}" for i, s in enumerate(interaction_types, start=1)}

    pivot_df["interaction_type_id"] = pivot_df["interaction_type"].map(interaction_map)
    data = pivot_df.to_dict(orient="list")

    source = ColumnDataSource(data=data)

    p3 = figure(
        x_axis_label="Opening question",
        y_axis_label="Success rate",
        height=250,
        width=400,
        output_backend="svg",
        x_range=data["interaction_type_id"],
        y_range=(0, 1)
    )

    bar_width = 0.8 / max(1, len(teacher_llms))

    for i, teacher_llm in enumerate(teacher_llms):
        offset = -bar_width * len(teacher_llms) / 2 + bar_width / 2 + i * bar_width
        p3.vbar(x=dodge("interaction_type_id", offset, range=p3.x_range), top=teacher_llm, source=source,
                width=bar_width, color=VISUAL_SETTINGS[teacher_llm]["color"])

    p3.x_range.range_padding = 0.1
    p3.xgrid.grid_line_color = None

    print()
    print("Opening question mapping")
    print(interaction_map)

    export_svgs(p3, filename="./figs/success_rate_by_opening_question_type.svg")

    #  ---- Success Rate by model size ----
    with_size = filtered_df[~np.isnan(filtered_df["teacher_llm_size"])]
    data = {"model_size": [], "success_rate": [], "upper": [], "lower": [], "color": []}
    for (teacher_llm, size), group in with_size.groupby(["teacher_llm", "teacher_llm_size"]):
        total = len(group)
        successes = group["assessment"].sum()
        success_rate = successes / total
        ci_low, ci_upp = smp.proportion_confint(successes, total, alpha=0.05, method='wilson')
        data["model_size"].append(size)
        data["success_rate"].append(success_rate)
        data["upper"].append(ci_upp)
        data["lower"].append(ci_low)
        data["color"].append(VISUAL_SETTINGS[teacher_llm]["color"])

    source = ColumnDataSource(data)

    p4 = figure(
        x_axis_type="log",
        y_range=(0, max(data["upper"]) + 0.1),
        x_axis_label="Model Size (billion parameters)",
        y_axis_label="Success Rate",
        height=250,
        width=400,
        output_backend="svg"
    )

    p4.scatter("model_size", "success_rate", source=source, size=10, color="color")
    p4.add_layout(Whisker(source=source, base="model_size",
                          upper="upper", lower="lower", line_color="color"))

    without_size = filtered_df[np.isnan(filtered_df["teacher_llm_size"])]
    for teacher_llm, group in without_size.groupby("teacher_llm"):
        total = len(group)
        successes = group["assessment"].sum()
        success_rate = successes / total
        hline = Span(
            location=success_rate,
            dimension='width',
            line_color=VISUAL_SETTINGS[teacher_llm]["color"],
            line_dash="dashed",
            line_width=2
        )
        p4.renderers.extend([hline, ])

    export_svgs(p4, filename="./figs/success_rate_by_model_size.svg")

    # Distribution of conversation length (early stopping)

    for teacher_llm, group in filtered_df.groupby("teacher_llm"):
        pX = figure(
            y_range=(0, 1.),
            title=f"{VISUAL_SETTINGS[teacher_llm]['display_name']}",
            x_axis_label="Conversation Rounds",
            y_axis_label="Probability Density",
            height=250,
            width=400,
            output_backend="svg"
        )

        hist, edges = np.histogram(
            (group["conversation_length"] - 1) / 2,
            density=True,
            bins=np.arange(1, df['max_interactions'].max() + 1)
        )

        edges = edges - 0.5

        pX.quad(
            top=hist,
            bottom=0,
            left=edges[:-1],
            right=edges[1:],
            fill_color=VISUAL_SETTINGS[teacher_llm]["color"],
            line_color="white"
        )

        safe_name = teacher_llm.replace(':', '_').replace('/', '_')
        export_svgs(pX, filename=f"./figs/distribution_conversation_length_{safe_name}.svg")
