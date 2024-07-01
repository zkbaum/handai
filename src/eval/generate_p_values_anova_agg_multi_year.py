import os
import sys
import pandas as pd
import numpy as np
from scipy.stats import f_oneway
from eval_util import get_chatgpt_df, Experiment

# Hack to import from parent dir
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
from private import ROOT_DIR

# Define the paths to the CSV files
years = [2009, 2010, 2011, 2012, 2013]
experiment_names = [
    "gpt3.5",
    "gpt4o",
    "gpt4o_better_prompt",
]


def load_datasets():
    datasets = {name: [] for name in experiment_names}
    for year in years:
        datasets["gpt3.5"].append(get_chatgpt_df(year, Experiment.GPT3_5))
        datasets["gpt4o"].append(get_chatgpt_df(year, Experiment.GPT4O))
        datasets["gpt4o_better_prompt"].append(
            get_chatgpt_df(year, Experiment.GPT4O_BETTER_PROMPT)
        )
    combined_datasets = {
        name: pd.concat(data) for name, data in datasets.items()
    }
    return combined_datasets


def _calculate_accuracy(answers_df):
    results = {q_type: [] for q_type in answers_df["question_type"].unique()}
    for qid, row in answers_df.iterrows():
        attempts = [row[f"attempt{n}"] for n in range(10)]
        correct_answer = row["actual_answer"]
        q_type = row["question_type"]
        results[q_type].extend(
            [1 if ans == correct_answer else 0 for ans in attempts]
        )
    accuracy = {q_type: np.mean(results[q_type]) for q_type in results}
    return accuracy, results


def check_assumptions_and_calculate_p_values(experiments, question_type):
    comparisons = []

    all_groups = []
    group_names = []

    for name, results in experiments:
        if question_type in results:
            group = results[question_type]
            if len(group) > 1:
                all_groups.append(group)
                group_names.append(name)

    n = len(all_groups)
    for i in range(n):
        for j in range(i + 1, n):
            name1, group1 = group_names[i], all_groups[i]
            name2, group2 = group_names[j], all_groups[j]
            if len(group1) > 1 and len(group2) > 1:
                f_val, p_val = f_oneway(group1, group2)
                label = "SIGNIFICANT" if p_val < 0.05 else "NOT_SIGNIFICANT"
                formatted_p_val = f"{p_val:.2e}"
                comparisons.append([name1, name2, formatted_p_val, label])

    return comparisons


def analyze_combined_data():
    datasets = load_datasets()
    results = [
        _calculate_accuracy(df.set_index("question_id"))[1]
        for df in datasets.values()
    ]
    experiments = list(zip(experiment_names, results))

    text_comparisons = check_assumptions_and_calculate_p_values(
        experiments, "Text"
    )
    image_comparisons = check_assumptions_and_calculate_p_values(
        experiments, "Image"
    )

    formatted_results = []
    for comparison in text_comparisons:
        formatted_results.append(["combined", "text"] + comparison)
    for comparison in image_comparisons:
        formatted_results.append(["combined", "image"] + comparison)

    return formatted_results


# Analyze combined data and compile the results
all_results = analyze_combined_data()

# Convert the results to a DataFrame and save to a CSV file
results_df = pd.DataFrame(
    all_results,
    columns=[
        "Year",
        "Type",
        "Experiment 1",
        "Experiment 2",
        "p-value",
        "verdict",
    ],
)
output_csv_path = f"{ROOT_DIR}/out/analysis/combined_p_values_analysis.csv"
results_df.to_csv(output_csv_path, index=False)

print(f"Combined results saved to {output_csv_path}")
