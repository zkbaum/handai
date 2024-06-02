import os
import sys
import pandas as pd
import numpy as np
from scipy.stats import f_oneway
from eval_util import get_chatgpt_df, get_key_df, get_human_df, Experiment

# Hack to import from parent dir
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
from private import ROOT_DIR

# Load datasets
key_df = get_key_df()
human_df = get_human_df()
gpt_3_5_df = get_chatgpt_df(Experiment.GPT3_5)
gpt4_df = get_chatgpt_df(Experiment.GPT4)
gpt4o_df = get_chatgpt_df(Experiment.GPT4O)
gpt4o_fewshot_df = get_chatgpt_df(Experiment.GPT4O_BETTER_PROMPT)
gpt4o_filesearch_df = get_chatgpt_df(Experiment.GPT4O_FILE_SEARCH)
gpt4o_filesearch_fewshot_df = get_chatgpt_df(
    Experiment.GPT4O_FILE_SEARCH_AND_BETTER_PROMPT
)

# Filter out Video questions from the key dataset
key_df = key_df[key_df["question_type"] != "Video"]
key_df.set_index("question_id", inplace=True)


def _calculate_accuracy(answers_df, key_df):
    results = {q_type: [] for q_type in key_df["question_type"].unique()}
    for qid, row in answers_df.iterrows():
        if qid in key_df.index:
            correct_answer = key_df.loc[qid, "correct_answer"]
            q_type = key_df.loc[qid, "question_type"]
            results[q_type].extend([1 if ans == correct_answer else 0 for ans in row])
    accuracy = {q_type: np.mean(results[q_type]) for q_type in results}
    return accuracy, results


# Calculate accuracies and raw results
human_accuracy, human_results = _calculate_accuracy(
    human_df.set_index("student_id").T, key_df
)
gpt_3_5_accuracy, gpt_3_5_results = _calculate_accuracy(
    gpt_3_5_df.set_index("question_id"), key_df
)
gpt4_accuracy, gpt4_results = _calculate_accuracy(
    gpt4_df.set_index("question_id"), key_df
)
gpt4o_accuracy, gpt4o_results = _calculate_accuracy(
    gpt4o_df.set_index("question_id"), key_df
)
gpt4ofewshot_accuracy, gpt4ofewshot_results = _calculate_accuracy(
    gpt4o_fewshot_df.set_index("question_id"), key_df
)
gpt4ofilesearch_accuracy, gpt4ofilesearch_results = _calculate_accuracy(
    gpt4o_filesearch_df.set_index("question_id"), key_df
)
gpt4ofilesearchfewshot_accuracy, gpt4ofilesearchfewshot_results = _calculate_accuracy(
    gpt4o_filesearch_fewshot_df.set_index("question_id"), key_df
)

# Create a list of results to compare
experiments = [
    # ("human", human_results),
    ("gpt3.5", gpt_3_5_results),
    ("gpt4", gpt4_results),
    ("gpt4o", gpt4o_results),
    ("gpt4o_fewshot", gpt4ofewshot_results),
    ("gpt4o_filesearch", gpt4ofilesearch_results),
    ("gpt4o_filesearch_fewshot", gpt4ofilesearchfewshot_results),
]

experiment_names = [exp[0] for exp in experiments]


# Function to calculate p-values using one-way ANOVA
def calculate_p_values_matrix(experiments, question_type):
    n = len(experiments)
    p_values_matrix = np.zeros((n, n))
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
    return p_values_matrix


# Calculate p-values matrices for Text and Image questions
text_p_values_matrix = calculate_p_values_matrix(experiments, "Text")
image_p_values_matrix = calculate_p_values_matrix(experiments, "Image")

# Convert p-values matrices to DataFrame for easier viewing
text_p_values_df = pd.DataFrame(
    text_p_values_matrix, index=experiment_names, columns=experiment_names
)
image_p_values_df = pd.DataFrame(
    image_p_values_matrix, index=experiment_names, columns=experiment_names
)

# Output the p-values matrices
print("Text p-values matrix:")
print(text_p_values_df)
print("\nImage p-values matrix:")
print(image_p_values_df)

# Optionally, save the dataframes to CSV files
text_output_path = f"{ROOT_DIR}/out/analysis/text_p_values_matrix.csv"
image_output_path = f"{ROOT_DIR}/out/analysis/image_p_values_matrix.csv"
text_p_values_df.to_csv(text_output_path)
image_p_values_df.to_csv(image_output_path)
print(f"wrote text output to {text_output_path}")
print(f"wrote image output to {image_output_path}")
