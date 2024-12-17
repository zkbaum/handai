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

# Define the years
years = [2009, 2010, 2011, 2012, 2013]


def _calculate_human_accuracy():
    accuracies = []
    for year in years:
        sample_df = get_chatgpt_df(year, Experiment.GPT4O)
        grouped_data = sample_df.groupby("question_type")["human_correct_percentage"]
        year_accuracy = {q_type: np.mean(data) / 100 for q_type, data in grouped_data}
        accuracies.append(year_accuracy)
    overall_accuracy = {
        "Text": np.mean([acc.get("Text", 0) for acc in accuracies]),
        "Image": np.mean([acc.get("Image", 0) for acc in accuracies]),
    }
    return overall_accuracy


def _calculate_chatgpt_accuracy_and_ci(experiment):
    results = {"Text": [], "Image": []}
    for year in years:
        answers_df = get_chatgpt_df(year, experiment)
        for _index, row in answers_df.iterrows():
            correct_answer = row["actual_answer"]
            q_type = row["question_type"]
            results[q_type].extend(
                [
                    1 if row[attempt] == correct_answer else 0
                    for attempt in [f"attempt{i}" for i in range(10)]
                ]
            )
    accuracy = {
        "Text": np.mean(results["Text"]),
        "Image": np.mean(results["Image"]),
    }
    ci = {
        q_type: sem(results[q_type]) * 1.96
        for q_type in results
        if len(results[q_type]) > 1
    }
    return accuracy, ci


def _calculate_question_counts():
    text_count = 0
    image_count = 0
    for year in years:
        sample_df = get_chatgpt_df(year, Experiment.GPT4O)
        text_count += sample_df[sample_df["question_type"] == "Text"].shape[0]
        image_count += sample_df[sample_df["question_type"] == "Image"].shape[0]
    return text_count, image_count


# Calculate accuracies, confidence intervals, and question counts
human_accuracy = _calculate_human_accuracy()
gpt_3_5_accuracy, gpt_3_5_ci = _calculate_chatgpt_accuracy_and_ci(Experiment.GPT3_5)
gpt4o_accuracy, gpt4o_ci = _calculate_chatgpt_accuracy_and_ci(Experiment.GPT4O)
gpt4ofewshot_accuracy, gpt4ofewshot_ci = _calculate_chatgpt_accuracy_and_ci(
    Experiment.GPT4O_BETTER_PROMPT
)
text_count, image_count = _calculate_question_counts()

# Bar chart plot
labels = [
    "Human",
    "GPT-3.5\n(Image not supported)",
    "GPT-4o",
    "GPT-4o with\nbetter prompt",
]
text_accuracies = [
    human_accuracy.get("Text", 0),
    gpt_3_5_accuracy.get("Text", 0),
    gpt4o_accuracy.get("Text", 0),
    gpt4ofewshot_accuracy.get("Text", 0),
]
image_accuracies = [
    human_accuracy.get("Image", 0),
    0,  # GPT-3.5 does not support image
    gpt4o_accuracy.get("Image", 0),
    gpt4ofewshot_accuracy.get("Image", 0),
]
text_ci = [
    0,  # No CI for human
    gpt_3_5_ci.get("Text", 0),
    gpt4o_ci.get("Text", 0),
    gpt4ofewshot_ci.get("Text", 0),
]
image_ci = [
    0,  # No CI for human
    0,  # GPT-3.5 does not support image
    gpt4o_ci.get("Image", 0),
    gpt4ofewshot_ci.get("Image", 0),
]

x = np.arange(len(labels))
width = 0.35

# Separate data with and without CI
text_accuracies_with_ci = [acc for acc, ci in zip(text_accuracies, text_ci) if ci > 0]
text_accuracies_without_ci = [
    acc for acc, ci in zip(text_accuracies, text_ci) if ci == 0
]
text_ci_with_ci = [ci for ci in text_ci if ci > 0]
x_with_ci = [i for i, ci in zip(x, text_ci) if ci > 0]
x_without_ci = [i for i, ci in zip(x, text_ci) if ci == 0]

image_accuracies_with_ci = [
    acc for acc, ci in zip(image_accuracies, image_ci) if ci > 0
]
image_accuracies_without_ci = [
    acc for acc, ci in zip(image_accuracies, image_ci) if ci == 0
]
image_ci_with_ci = [ci for ci in image_ci if ci > 0]
x_image_with_ci = [i for i, ci in zip(x, image_ci) if ci > 0]
x_image_without_ci = [i for i, ci in zip(x, image_ci) if ci == 0]

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

# Define appealing colors for text and image bars
text_color = "#1f77b4"  # Deep Blue
image_color = "#ff7f0e"  # Orange

# Increase figsize and dpi
fig, ax = plt.subplots(figsize=(15, 10), dpi=100)

# Plot bars with CI
rects1_with_ci = ax.bar(
    [i - width / 2 for i in x_with_ci],
    text_accuracies_with_ci,
    width,
    color=text_color,
    yerr=text_ci_with_ci,
    capsize=5,
    error_kw={"elinewidth": 1, "alpha": 0.5},
)

rects2_with_ci = ax.bar(
    [i + width / 2 for i in x_image_with_ci],
    image_accuracies_with_ci,
    width,
    color=image_color,
    yerr=image_ci_with_ci,
    capsize=5,
    error_kw={"elinewidth": 1, "alpha": 0.5},
)

# Plot bars without CI
rects1_without_ci = ax.bar(
    [i - width / 2 for i in x_without_ci],
    text_accuracies_without_ci,
    width,
    color=text_color,
    capsize=5,
)

rects2_without_ci = ax.bar(
    [i + width / 2 for i in x_image_without_ci],
    image_accuracies_without_ci,
    width,
    color=image_color,
    capsize=5,
)

ax.set_ylim([0, 1])
ax.set_xlabel("Group")
ax.set_ylabel("Accuracy")
ax.set_title("Performance of humans vs ChatGPT on 5 years of self-assessment exams")
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=12)

# Manually create legend with number of questions
ax.legend(
    [rects1_with_ci, rects2_with_ci],
    [f"Text questions (N={text_count})", f"Image questions (N={image_count})"],
    loc="upper right",
)


def _autolabel(rects, ci):
    """
    Add text labels for accuracy above ci
    """
    for rect, c in zip(rects, ci):
        height = rect.get_height()
        label = f"{height:.1%}" if c == 0 else f"{height:.1%} ± {c:.1%}"
        if height == 0:
            label = ""
        ax.annotate(
            label,
            xy=(rect.get_x() + rect.get_width() / 2, height),
            xytext=(0, 5 + c * 500),  # Increased offset to prevent overlap
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=12,
        )


# Add labels to bars with CI
_autolabel(rects1_with_ci, text_ci_with_ci)
_autolabel(rects2_with_ci, image_ci_with_ci)

# Add labels to bars without CI
_autolabel(rects1_without_ci, [0] * len(rects1_without_ci))
_autolabel(rects2_without_ci, [0] * len(rects2_without_ci))

plt.tight_layout()
out_path = f"{ROOT_DIR}/out/analysis/graph_with_ci_2009_2013.png"
plt.savefig(out_path, dpi=300)  # Save the plot with high DPI
print(f"Wrote graph to {out_path}")
