import pandas as pd
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import sem
from eval_util import get_chatgpt_df, Experiment

# Hack to import from parent dir
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
from private import ROOT_DIR

print("working...this may take a few seconds")


def _get_n_per_year(year: int):
    ns = {
        2008: 1706,
        2009: 1845,
        2010: 1832,
        2011: 1959,
        2012: 1150,
        2013: 1721,
    }
    return ns[year]


def _get_question_count(year: int):
    sample_df = get_chatgpt_df(year, Experiment.GPT4O)
    text_count = sample_df[sample_df["question_type"] == "Text"].shape[0]
    image_count = sample_df[sample_df["question_type"] == "Image"].shape[0]
    return text_count, image_count


def _calculate_human_accuracy_and_ci(year: int):
    n = _get_n_per_year(year)
    sample_df = get_chatgpt_df(year, Experiment.GPT4O)
    grouped_data = sample_df.groupby("question_type")["human_correct_percentage"]
    accuracy = {q_type: np.mean(data) / 100 for q_type, data in grouped_data}
    std_err_corrected = grouped_data.std() / np.sqrt(n)
    ci_corrected = std_err_corrected * 1.96 / 100
    ci = {q_type: ci_corrected[q_type] for q_type in ci_corrected.index}
    return accuracy, ci


def _calculate_chatgpt_accuracy_and_ci(answers_df):
    results = {q_type: [] for q_type in answers_df["question_type"].unique()}
    for _index, row in answers_df.iterrows():
        correct_answer = row["actual_answer"]
        q_type = row["question_type"]
        results[q_type].extend(
            [
                1 if row[attempt] == correct_answer else 0
                for attempt in [
                    "attempt0",
                    "attempt1",
                    "attempt2",
                    "attempt3",
                    "attempt4",
                    "attempt5",
                    "attempt6",
                    "attempt7",
                    "attempt8",
                    "attempt9",
                ]
            ]
        )
    accuracy = {q_type: np.mean(results[q_type]) for q_type in results}
    ci = {
        q_type: sem(results[q_type]) * 1.96
        for q_type in results
        if len(results[q_type]) > 1
    }
    return accuracy, ci


def process_year(year):
    n = _get_n_per_year(year)
    text_count, image_count = _get_question_count(year)
    human_accuracy, human_ci = _calculate_human_accuracy_and_ci(year)
    gpt_3_5_df = get_chatgpt_df(year, Experiment.GPT3_5)
    gpt4o_df = get_chatgpt_df(year, Experiment.GPT4O)
    gpt4o_better_prompt_df = get_chatgpt_df(year, Experiment.GPT4O_BETTER_PROMPT)
    gpt_3_5_accuracy, gpt_3_5_ci = _calculate_chatgpt_accuracy_and_ci(gpt_3_5_df)
    gpt4o_accuracy, gpt4o_ci = _calculate_chatgpt_accuracy_and_ci(gpt4o_df)
    gpt4o_better_prompt_accuracy, gpt4o_better_prompt_ci = (
        _calculate_chatgpt_accuracy_and_ci(gpt4o_better_prompt_df)
    )

    group_name = f"{year}\n{n} humans\n{text_count} text questions\n{image_count} image questions"

    labels = [
        "human text",
        "human image",
        "gpt3.5 text",
        "gpt4o text",
        "gpt4o image",
        "gpt4o better prompt text",
        "gpt4o better prompt image",
    ]

    accuracies = [
        human_accuracy.get("Text", 0),
        human_accuracy.get("Image", 0),
        gpt_3_5_accuracy.get("Text", 0),
        gpt4o_accuracy.get("Text", 0),
        gpt4o_accuracy.get("Image", 0),
        gpt4o_better_prompt_accuracy.get("Text", 0),
        gpt4o_better_prompt_accuracy.get("Image", 0),
    ]

    ci = [
        human_ci.get("Text", 0),
        human_ci.get("Image", 0),
        gpt_3_5_ci.get("Text", 0),
        gpt4o_ci.get("Text", 0),
        gpt4o_ci.get("Image", 0),
        gpt4o_better_prompt_ci.get("Text", 0),
        gpt4o_better_prompt_ci.get("Image", 0),
    ]

    return group_name, labels, accuracies, ci


years = [2009, 2010, 2011, 2012, 2013]
group_names = []
all_labels = []
all_accuracies = []
all_ci = []

for year in years:
    group_name, labels, accuracies, ci = process_year(year)
    group_names.append(group_name)
    all_labels.extend(labels)
    all_accuracies.extend(accuracies)
    all_ci.extend(ci)

num_bars_per_year = len(all_labels) // len(years)
spacing = 3  # Adjust this value to add more space between groups

# Generate x positions with appropriate spacing
x = np.array([])
for i in range(len(years)):
    year_x = np.arange(
        i * (num_bars_per_year + spacing),
        i * (num_bars_per_year + spacing) + num_bars_per_year,
    )
    x = np.concatenate([x, year_x])

# Define colors for each category
colors = [
    "#1f77b4",
    "#aec7e8",  # human text, human image
    "#d62728",
    "#2ca02c",
    "#98df8a",  # gpt3.5 text, gpt4o text, gpt4o image
    "#9467bd",
    "#c5b0d5",  # gpt4o better prompt text, gpt4o better prompt image
] * len(years)

fig, ax = plt.subplots(figsize=(18, 10), dpi=300)  # Increase DPI here
rects = ax.bar(
    x,  # Ensure x has the same length as all_accuracies
    all_accuracies,
    yerr=all_ci,
    capsize=5,
    error_kw={"elinewidth": 1, "alpha": 0.5},
    color=colors,
)

ax.set_ylim([0, 1])
ax.set_xlabel("\nYear\n(ChatGPT evaluated 10 times per year)")
ax.set_ylabel("Accuracy")
# TODO(zkbaum) rename and then send :)
ax.set_title("Performance of humans vs ChatGPT on 5 years of self-assessment exams")
ax.set_xticks(
    [
        i * (num_bars_per_year + spacing) + (num_bars_per_year - 1) / 2
        for i in range(len(years))
    ]
)
ax.set_xticklabels(group_names, rotation=0)

# Define legend entries
legend_entries = [
    ("human", ["text", "image"], ["#1f77b4", "#aec7e8"]),
    # empty string needs to be there, this is a hack to get gpt3.5
    # on its own column
    ("gpt3.5", ["text", ""], ["#d62728", "#FFFFFF"]),
    ("gpt4o", ["text", "image"], ["#2ca02c", "#98df8a"]),
    ("gpt4o better prompt", ["text", "image"], ["#9467bd", "#c5b0d5"]),
]

handles = []
labels = []
for group, types, group_colors in legend_entries:
    for typ, color in zip(types, group_colors):
        handles.append(plt.Rectangle((0, 0), 1, 1, color=color))
        # hack to get gpt3.5 on its own column
        if typ == "":
            group = ""
        labels.append(f"{group} {typ}")

# Arrange legend in 4 columns and place it at the default location
ax.legend(handles, labels, loc="best", ncol=4)


def _autolabel(rects, ci):
    """
    Add text labels for accuracy and CI rotated 90 degrees and placed below the lowest error bar
    """
    for rect, c in zip(rects, ci):
        height = rect.get_height()
        if height != 0:
            ax.annotate(
                f"{height * 100:.1f}% ± {c * 100:.2f}%",
                xy=(rect.get_x() + rect.get_width() / 2, height - c - 0.05),
                xytext=(
                    0,
                    20,
                ),  # Adjust this value to place text further below the lowest error bar
                textcoords="offset points",
                ha="center",
                va="top",
                rotation=90,
                fontsize=8,
                # weight="bold",
            )


_autolabel(rects, all_ci)

plt.tight_layout()
out_path = f"{ROOT_DIR}/out/analysis/graph_with_ci_multi_year.png"
plt.savefig(out_path, dpi=300)  # Save the plot with high DPI
print(f"Wrote graph to {out_path}")
# plt.show()
