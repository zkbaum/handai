import os
import sys
import pandas as pd
import numpy as np
from scipy.stats import f_oneway, shapiro, levene
from eval_util import get_chatgpt_df, Experiment

# Hack to import from parent dir
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
from private import ROOT_DIR

# Load datasets
gpt_3_5_df = get_chatgpt_df(2013000, Experiment.GPT3_5)
gpt4_df = get_chatgpt_df(2013000, Experiment.GPT4)
gpt4o_df = get_chatgpt_df(2013000, Experiment.GPT4O)
gpt4o_fewshot_df = get_chatgpt_df(2013000, Experiment.GPT4O_BETTER_PROMPT)
gpt4o_filesearch_df = get_chatgpt_df(2013000, Experiment.GPT4O_FILE_SEARCH)
gpt4o_filesearch_fewshot_df = get_chatgpt_df(
    2013000, Experiment.GPT4O_FILE_SEARCH_AND_BETTER_PROMPT
)


def _calculate_accuracy(answers_df):
    results = {q_type: [] for q_type in answers_df["question_type"].unique()}
    for qid, row in answers_df.iterrows():
        attempts = [row[f"attempt{n}"] for n in range(10)]
        correct_answer = row["actual_answer"]
        q_type = row["question_type"]
        results[q_type].extend([1 if ans == correct_answer else 0 for ans in attempts])
    accuracy = {q_type: np.mean(results[q_type]) for q_type in results}
    return accuracy, results


# Calculate accuracies and raw results
gpt_3_5_accuracy, gpt_3_5_results = _calculate_accuracy(
    gpt_3_5_df.set_index("question_id")
)
gpt4_accuracy, gpt4_results = _calculate_accuracy(gpt4_df.set_index("question_id"))
gpt4o_accuracy, gpt4o_results = _calculate_accuracy(gpt4o_df.set_index("question_id"))
gpt4ofewshot_accuracy, gpt4ofewshot_results = _calculate_accuracy(
    gpt4o_fewshot_df.set_index("question_id")
)
gpt4ofilesearch_accuracy, gpt4ofilesearch_results = _calculate_accuracy(
    gpt4o_filesearch_df.set_index("question_id")
)
gpt4ofilesearchfewshot_accuracy, gpt4ofilesearchfewshot_results = _calculate_accuracy(
    gpt4o_filesearch_fewshot_df.set_index("question_id")
)

# Create a list of results to compare
experiments = [
    # do not include human because the sample sizes are so different
    # ("human", human_results),
    ("gpt3.5", gpt_3_5_results),
    ("gpt4", gpt4_results),
    ("gpt4o", gpt4o_results),
    ("gpt4o_fewshot", gpt4ofewshot_results),
    ("gpt4o_filesearch", gpt4ofilesearch_results),
    ("gpt4o_filesearch_fewshot", gpt4ofilesearchfewshot_results),
]

experiment_names = [exp[0] for exp in experiments]


# Function to check assumptions and calculate p-values using one-way ANOVA
def check_assumptions_and_calculate_p_values(experiments, question_type):
    normality_results = {}
    homogeneity_results = {}
    n = len(experiments)
    p_values_matrix = np.zeros((n, n))

    # Collect groups for Levene's test
    all_groups = []

    for name, results in experiments:
        if question_type in results:
            group = results[question_type]
            if len(group) > 1:  # Ensure at least 2 samples per group
                all_groups.append(group)
                # Check normality
                stat, p_value = shapiro(group)
                normality_results[name] = (stat, p_value)

    # Check homogeneity of variances
    if len(all_groups) > 1:
        stat, p_value = levene(*all_groups)
        homogeneity_results["Levene's Test"] = (stat, p_value)

    # Calculate p-values for ANOVA
    for i in range(n):
        for j in range(i + 1, n):
            name1, results1 = experiments[i]
            name2, results2 = experiments[j]
            if question_type in results1 and question_type in results2:
                group1 = results1[question_type]
                group2 = results2[question_type]
                if (
                    len(group1) > 1 and len(group2) > 1
                ):  # ANOVA requires at least 2 samples per group
                    f_val, p_val = f_oneway(group1, group2)
                    p_values_matrix[i, j] = p_val
                    p_values_matrix[j, i] = p_val  # Symmetric matrix

    return p_values_matrix, normality_results, homogeneity_results


# Calculate p-values matrices and check assumptions for Text and Image questions
text_p_values_matrix, text_normality, text_homogeneity = (
    check_assumptions_and_calculate_p_values(experiments, "Text")
)
image_p_values_matrix, image_normality, image_homogeneity = (
    check_assumptions_and_calculate_p_values(experiments, "Image")
)

# Convert p-values matrices to DataFrame for easier viewing
text_p_values_df = pd.DataFrame(
    text_p_values_matrix, index=experiment_names, columns=experiment_names
)
image_p_values_df = pd.DataFrame(
    image_p_values_matrix, index=experiment_names, columns=experiment_names
)

# Output the p-values matrices and assumptions checks
print("Text p-values matrix:")
print(text_p_values_df)
print("\nText Normality Results:")
for exp_name, (stat, p_val) in text_normality.items():
    print(f"{exp_name}: Stat={stat}, p-value={p_val}")
print("\nText Homogeneity Result:")
for test_name, (stat, p_val) in text_homogeneity.items():
    print(f"{test_name}: Stat={stat}, p-value={p_val}")

print("\nImage p-values matrix:")
print(image_p_values_df)
print("\nImage Normality Results:")
for exp_name, (stat, p_val) in image_normality.items():
    print(f"{exp_name}: Stat={stat}, p-value={p_val}")
print("\nImage Homogeneity Result:")
for test_name, (stat, p_val) in image_homogeneity.items():
    print(f"{test_name}: Stat={stat}, p-value={p_val}")


# Optionally, save the dataframes to CSV files
text_output_path = f"{ROOT_DIR}/out/analysis/text_p_values_matrix_anova.csv"
image_output_path = f"{ROOT_DIR}/out/analysis/image_p_values_matrix_anova.csv"
text_p_values_df.to_csv(text_output_path)
image_p_values_df.to_csv(image_output_path)
print(f"wrote text output to {text_output_path}")
print(f"wrote image output to {image_output_path}")

# now convert to a friendlier format


# Function to determine the verdict
def get_verdict(p_value):
    return "SIGNIFICANT" if p_value < 0.05 else "NOT_SIGNIFICANT"


# Create results DataFrame in the specified format
results = []

# Perform comparisons for text and image dataframes
for df, df_type in zip([text_p_values_df, image_p_values_df], ["text", "image"]):
    for i in range(len(df.index)):
        for j in range(i + 1, len(df.columns)):
            p_value = df.iloc[i, j]
            if p_value != 0:  # Exclude comparisons where p_value is 0
                results.append(
                    [df_type, df.index[i], df.columns[j], p_value, get_verdict(p_value)]
                )

final_df = pd.DataFrame(
    results, columns=["Type", "Experiment 1", "Experiment 2", "p-value", "verdict"]
)

final_output_path = f"{ROOT_DIR}/out/analysis/final_anova.csv"
final_df.to_csv(final_output_path)
print(f"wrote final combined output to {final_output_path}")
