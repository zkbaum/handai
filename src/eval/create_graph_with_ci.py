"""
Generates a graph of accuracy per experiment, with confidence intervals.
Stratified by Text and Image.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import sem
from eval_util import get_chatgpt_df, get_key_df, get_human_df, Experiment

# Load datasets

# CSV in the form [question_id, correct_answer, question_type]
key_df = get_key_df()

# CSV in the form [student_id, answer1, answer2, ..., answer200]
human_df = get_human_df()

# CSV in the form [question_id, attempt0, attempt1, attempt2]
gpt_3_5_df = get_chatgpt_df(Experiment.ZERO_SHOT_GPT3_5)
gpt4_df = get_chatgpt_df(Experiment.ZERO_SHOT_GPT4)
gpt4o_df = get_chatgpt_df(Experiment.ZERO_SHOT_GPT4O)
gpt4o_fewshot_df = get_chatgpt_df(Experiment.FEW_SHOT_GPT4O)
gpt4o_filesearch_df = get_chatgpt_df(Experiment.FILE_SEARCH_ZERO_SHOT)
gpt4o_filesearch_fewshot_df = get_chatgpt_df(Experiment.FILE_SEARCH_FEW_SHOT)


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
human_accuracy, human_ci = _calculate_human_accuracy_and_ci(human_df, key_df)
gpt_3_5_accuracy, gpt_3_5_ci = _calculate_chatgpt_accuracy_and_ci(gpt_3_5_df, key_df)
gpt4_accuracy, gpt4_ci = _calculate_chatgpt_accuracy_and_ci(gpt4_df, key_df)
gpt4o_accuracy, gpt4o_ci = _calculate_chatgpt_accuracy_and_ci(gpt4o_df, key_df)
gpt4ofewshot_accuracy, gpt4ofewshot_ci = _calculate_chatgpt_accuracy_and_ci(
    gpt4o_fewshot_df, key_df
)
gpt4ofilesearch_accuracy, gpt4ofilesearch_ci = _calculate_chatgpt_accuracy_and_ci(
    gpt4o_filesearch_df, key_df
)
gpt4ofilesearchfewshot_accuracy, gpt4ofilesearchfewshot_ci = (
    _calculate_chatgpt_accuracy_and_ci(gpt4o_filesearch_fewshot_df, key_df)
)


# Bar chart plot
labels = [
    "human ",
    "gpt3.5",
    "gpt4",
    "gpt4o",
    "gpt4o with\nbetter prompt",
    "gpt4o with\nfile search",
    "gpt4o with \nfile search and\nbetter prompt",
]
text_accuracies = [
    human_accuracy.get("Text", 0),
    gpt_3_5_accuracy.get("Text", 0),
    gpt4_accuracy.get("Text", 0),
    gpt4o_accuracy.get("Text", 0),
    gpt4ofewshot_accuracy.get("Text", 0),
    gpt4ofilesearch_accuracy.get("Text", 0),
    gpt4ofilesearchfewshot_accuracy.get("Text", 0),
]
image_accuracies = [
    human_accuracy.get("Image", 0),
    0,  # gpt3.5 does not support image
    gpt4_accuracy.get("Image", 0),
    gpt4o_accuracy.get("Image", 0),
    gpt4ofewshot_accuracy.get("Image", 0),
    0,  # file search does not support image
    0,  # file search does not support image
]
text_ci = [
    human_ci.get("Text", 0),
    gpt_3_5_ci.get("Text", 0),
    gpt4_ci.get("Text", 0),
    gpt4o_ci.get("Text", 0),
    gpt4ofewshot_ci.get("Text", 0),
    gpt4ofilesearch_ci.get("Text", 0),
    gpt4ofilesearchfewshot_ci.get("Text", 0),
]
image_ci = [
    human_ci.get("Image", 0),
    0,  # gpt3.5 does not support image
    gpt4_ci.get("Image", 0),
    gpt4o_ci.get("Image", 0),
    gpt4ofewshot_ci.get("Image", 0),
    0,  # file search does not support image
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
    error_kw={"elinewidth": 1, "alpha": 0.5},
)
rects2 = ax.bar(
    x + width / 2,
    image_accuracies,
    width,
    label="Image (N=51)",
    yerr=image_ci,
    capsize=5,
    error_kw={"elinewidth": 1, "alpha": 0.5},
)

ax.set_ylim([0, 1])
ax.set_xlabel("Group")
ax.set_ylabel("Accuracy (%)")
ax.set_title("Performance of humans vs ChatGPT on 2013 self-assessment")
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

# Add horizontal lines for human bars with matching colors
# human_accuracy_text = human_accuracy.get("Text", 0)
# human_accuracy_image = human_accuracy.get("Image", 0)
# ax.axhline(human_accuracy_text, color="blue", linestyle="--", linewidth=1)
# ax.axhline(human_accuracy_image, color="orange", linestyle="--", linewidth=1)


def _autolabel(rects, ci):
    """
    Add text labels for accuracy above ci
    """
    for rect, c in zip(rects, ci):
        height = rect.get_height()
        if height != 0:
            ax.annotate(
                f"{height:.2%}",
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 15),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
                weight="bold",
            )
            ax.annotate(
                f"±{c:.2%}",
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 5),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
            )


_autolabel(rects1, text_ci)
_autolabel(rects2, image_ci)

plt.tight_layout()
plt.show()
