"""
Generates a graph of accuracy per experiment, with confidence intervals.
Stratified by Text and Image.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import sem
from private import get_stats_csv_rootdir

# Load datasets
ROOT_DIR = get_stats_csv_rootdir()

# CSV in the form [question_id, correct_answer, question_type]
key_df = pd.read_csv(f"{ROOT_DIR}/key.csv")

# CSV in the form [student_id, answer1, answer2, ..., answer200]
human_df = pd.read_csv(f"{ROOT_DIR}/human.csv")

# CSV in the form [question_id, attempt0, attempt1, attempt2]
gpt_3_5_df = pd.read_csv(f"{ROOT_DIR}/gpt3-5.csv")
gpt4_df = pd.read_csv(f"{ROOT_DIR}/gpt4.csv")
gpt4_fewshot_df = pd.read_csv(f"{ROOT_DIR}/gpt4-fewshot.csv")
gpt4_filesearch_df = pd.read_csv(f"{ROOT_DIR}/gpt4-filesearch.csv")


# Filter out Video questions from the key dataset
key_df = key_df[key_df["question_type"] != "Video"]
key_df.set_index("question_id", inplace=True)


def _calculate_human_accuracy_and_ci(answers_df, key_df):
    results = {q_type: [] for q_type in key_df["question_type"].unique()}
    answers_df = answers_df.set_index("student_id").T  # Transpose for easier iteration
    for qid, answers in answers_df.iterrows():
        if qid in key_df.index:
            correct_answer = key_df.loc[qid, "correct_answer"]
            q_type = key_df.loc[qid, "question_type"]
            results[q_type].extend(
                [1 if ans == correct_answer else 0 for ans in answers]
            )
    accuracy = {q_type: np.mean(results[q_type]) for q_type in results}
    ci = {
        q_type: sem(results[q_type]) * 1.96
        for q_type in results
        if len(results[q_type]) > 1
    }
    return accuracy, ci


def _calculate_chatgpt_accuracy_and_ci(answers_df, key_df):
    results = {q_type: [] for q_type in key_df["question_type"].unique()}
    for _index, row in answers_df.iterrows():
        qid = row["question_id"]
        if qid in key_df.index:
            correct_answer = key_df.loc[qid, "correct_answer"]
            q_type = key_df.loc[qid, "question_type"]
            results[q_type].extend(
                [
                    1 if row[attempt] == correct_answer else 0
                    for attempt in [
                        "attempt0",
                        "attempt1",
                        "attempt2",
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
human_accuracy, human_ci = _calculate_human_accuracy_and_ci(human_df, key_df)
gpt_3_5_accuracy, gpt_3_5_ci = _calculate_chatgpt_accuracy_and_ci(gpt_3_5_df, key_df)
gpt4_accuracy, gpt4_ci = _calculate_chatgpt_accuracy_and_ci(gpt4_df, key_df)
gpt4fewshot_accuracy, gpt4fewshot_ci = _calculate_chatgpt_accuracy_and_ci(
    gpt4_fewshot_df, key_df
)
gpt4filesearch_accuracy, gpt4filesearch_ci = _calculate_chatgpt_accuracy_and_ci(
    gpt4_filesearch_df, key_df
)


# Bar chart plot
labels = [
    "human ",
    "gpt3.5",
    "gpt4",
    "gpt4-fewshot",
    "gpt4-filesearch",
]
text_accuracies = [
    human_accuracy.get("Text", 0),
    gpt_3_5_accuracy.get("Text", 0),
    gpt4_accuracy.get("Text", 0),
    gpt4fewshot_accuracy.get("Text", 0),
    gpt4filesearch_accuracy.get("Text", 0),
]
image_accuracies = [
    human_accuracy.get("Image", 0),
    0,  # gpt3.5 does not support image
    gpt4_accuracy.get("Image", 0),
    gpt4fewshot_accuracy.get("Image", 0),
    0,  # file search does not support image
]
text_ci = [
    human_ci.get("Text", 0),
    gpt_3_5_ci.get("Text", 0),
    gpt4_ci.get("Text", 0),
    gpt4fewshot_ci.get("Text", 0),
    gpt4filesearch_ci.get("Text", 0),
]
image_ci = [
    human_ci.get("Image", 0),
    0,  # gpt3.5 does not support image
    gpt4_ci.get("Image", 0),
    gpt4fewshot_ci.get("Image", 0),
    0,  # file search does not support image
]

x = np.arange(len(labels))
width = 0.35

fig, ax = plt.subplots()
rects1 = ax.bar(
    x - width / 2,
    text_accuracies,
    width,
    label="Text (N=144)",
    yerr=text_ci,
    capsize=5,
)
rects2 = ax.bar(
    x + width / 2,
    image_accuracies,
    width,
    label="Image (N=51)",
    yerr=image_ci,
    capsize=5,
)

ax.set_ylim([0, 1])
ax.set_xlabel("Group")
ax.set_ylabel("Accuracy (%)")
ax.set_title("Performance of humans vs ChatGPT on 2013 self-assessment")
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()


def _autolabel(rects, ci):
    """
    Add text labels for accuracy and ci (to be included above error bars)
    """
    for rect, c in zip(rects, ci):
        height = rect.get_height()
        if height != 0:
            vertical_offset = 0
            if c > 0.015:
                vertical_offset += 15
            if c > 0.05:
                vertical_offset += 15
            ax.annotate(
                f"{height:.2%}",
                xy=(rect.get_x() + rect.get_width() / 2, height + 0.01),
                xytext=(0, 10 + vertical_offset),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
            )
            ax.annotate(
                f"±{c:.2%}",
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 4 + vertical_offset),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=6,
            )


_autolabel(rects1, text_ci)
_autolabel(rects2, image_ci)

plt.tight_layout()
plt.show()
