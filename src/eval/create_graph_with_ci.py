import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import sem
from eval_util import get_chatgpt_df, Experiment

# Hack to import from parent dir
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
from private import ROOT_DIR

print("working...this may take a few seconds")

# Load datasets
# CSV in the form [question_id, attempt0, attempt1, attempt2]
gpt_3_5_df = get_chatgpt_df(2013000, Experiment.GPT3_5)
# gpt4_df = get_chatgpt_df(2013000, Experiment.GPT4)
gpt4o_df = get_chatgpt_df(2013000, Experiment.GPT4O)
gpt4o_fewshot_df = get_chatgpt_df(2013000, Experiment.GPT4O_BETTER_PROMPT)
gpt4o_filesearch_df = get_chatgpt_df(2013000, Experiment.GPT4O_FILE_SEARCH)
gpt4o_filesearch_fewshot_df = get_chatgpt_df(
    2013000, Experiment.GPT4O_FILE_SEARCH_AND_BETTER_PROMPT
)

NUM_HUMAN_EXAMINEES = 1721


def _calculate_human_accuracy_and_ci():
    sample_df = get_chatgpt_df(2013, Experiment.GPT4O)
    grouped_data = sample_df.groupby("question_type")["human_correct_percentage"]
    accuracy = {q_type: np.mean(data) / 100 for q_type, data in grouped_data}
    std_err_corrected = grouped_data.std() / np.sqrt(NUM_HUMAN_EXAMINEES)
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


# Calculate accuracies and confidence intervals
human_accuracy, human_ci = _calculate_human_accuracy_and_ci()
# gpt_3_5_accuracy, gpt_3_5_ci = _calculate_chatgpt_accuracy_and_ci(gpt_3_5_df)
# gpt4_accuracy, gpt4_ci = _calculate_chatgpt_accuracy_and_ci(gpt4_df)
gpt4o_accuracy, gpt4o_ci = _calculate_chatgpt_accuracy_and_ci(gpt4o_df)
gpt4ofewshot_accuracy, gpt4ofewshot_ci = _calculate_chatgpt_accuracy_and_ci(
    gpt4o_fewshot_df
)
gpt4ofilesearch_accuracy, gpt4ofilesearch_ci = _calculate_chatgpt_accuracy_and_ci(
    gpt4o_filesearch_df
)
# gpt4ofilesearchfewshot_accuracy, gpt4ofilesearchfewshot_ci = (
#     _calculate_chatgpt_accuracy_and_ci(gpt4o_filesearch_fewshot_df)
# )

# Bar chart plot
labels = [
    f"Human\n(on {NUM_HUMAN_EXAMINEES} examinees)",
    # "gpt3.5 - image\n not supported\n(run 10 times)",
    # "gpt4",
    "ChatGPT 4o\n(experiment 2)",
    "ChatGPT 4o with\nbetter prompt\n(experiment 3)",
    "ChatGPT 4o with\nfile search\n(experiment 4)",
    # "gpt4o with\nfile search and\nbetter prompt",
]
text_accuracies = [
    human_accuracy.get("Text", 0),
    # gpt_3_5_accuracy.get("Text", 0),
    # gpt4_accuracy.get("Text", 0),
    gpt4o_accuracy.get("Text", 0),
    gpt4ofewshot_accuracy.get("Text", 0),
    gpt4ofilesearch_accuracy.get("Text", 0),
    # gpt4ofilesearchfewshot_accuracy.get("Text", 0),
]
image_accuracies = [
    human_accuracy.get("Image", 0),
    # 0,  # gpt3.5 does not support image
    # gpt4_accuracy.get("Image", 0),
    gpt4o_accuracy.get("Image", 0),
    gpt4ofewshot_accuracy.get("Image", 0),
    gpt4ofilesearch_accuracy.get("Image", 0),
    # gpt4ofilesearchfewshot_accuracy.get("Image", 0),
]
text_ci = [
    human_ci.get("Text", 0),
    # gpt_3_5_ci.get("Text", 0),
    # gpt4_ci.get("Text", 0),
    gpt4o_ci.get("Text", 0),
    gpt4ofewshot_ci.get("Text", 0),
    gpt4ofilesearch_ci.get("Text", 0),
    # gpt4ofilesearchfewshot_ci.get("Text", 0),
]
image_ci = [
    human_ci.get("Image", 0),
    # 0,  # gpt3.5 does not support image
    # gpt4_ci.get("Image", 0),
    gpt4o_ci.get("Image", 0),
    gpt4ofewshot_ci.get("Image", 0),
    gpt4ofilesearch_ci.get("Image", 0),
    # gpt4ofilesearchfewshot_ci.get("Image", 0),
]

x = np.arange(len(labels))
width = 0.35

# Increase font sizes
plt.rcParams.update(
    {
        "axes.titlesize": 16,
        "axes.labelsize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
    }
)

# Increase figsize and dpi
fig, ax = plt.subplots(figsize=(15, 10), dpi=100)
rects1 = ax.bar(
    x - width / 2,
    text_accuracies,
    width,
    label="Text questions (n=144)",
    yerr=text_ci,
    capsize=5,
    error_kw={"elinewidth": 1, "alpha": 0.5},
)
rects2 = ax.bar(
    x + width / 2,
    image_accuracies,
    width,
    label="Image questions (n=51)",
    yerr=image_ci,
    capsize=5,
    error_kw={"elinewidth": 1, "alpha": 0.5},
)

ax.set_ylim([0, 1])
ax.set_xlabel("Group")
ax.set_ylabel("Accuracy")
ax.set_title("Performance of Humans vs ChatGPT on self-assessment exam (2013 only)")
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=12)
ax.legend()


def _autolabel(rects, ci):
    """
    Add text labels for accuracy above ci
    """
    for rect, c in zip(rects, ci):
        height = rect.get_height()
        if height != 0:
            # Adjust label position to be just above the error bar
            error_bar_height = c if isinstance(c, float) else 0
            ax.annotate(
                f"{height:.1%} ± {c:.1%}",
                xy=(
                    rect.get_x() + rect.get_width() / 2,
                    height + error_bar_height,
                ),
                xytext=(0, 5),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=10,
                # weight="bold",
            )


_autolabel(rects1, text_ci)
_autolabel(rects2, image_ci)

plt.tight_layout()
# plt.show()

out_path = f"{ROOT_DIR}/out/analysis/graph_with_ci_2013.png"
plt.savefig(out_path, dpi=300)  # Save the plot with high DPI
print(f"Wrote graph to {out_path}")
