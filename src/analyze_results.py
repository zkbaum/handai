"""
Script to analyze results from a hand AI experiment.
"""

import pandas as pd
from enum import Enum
from private import get_result_csvpath_for_experiment, ROOT_DIR
import csv


class Experiment(Enum):
    HUMAN_CONTROL = "human"
    ZERO_SHOT_GPT3_5 = "zero-shot-gpt3.5"
    ZERO_SHOT_GPT4 = "zero-shot-gpt4"
    ZERO_SHOT_GPT4O = "zero-shot-gpt4o"
    FEW_SHOT_GPT4O = "few-shot-gpt4o"
    FILE_SEARCH_ZERO_SHOT = "rag-zero-shot"
    FILE_SEARCH_FEW_SHOT = "rag-few-shot"


def _determine_majority_or_tie(row):
    """
    Determines the most frequent answer, or "TIE" if there's a tie.
    """
    modes = row.mode()
    if modes.empty:  # Check if the modes Series is empty
        return None  # Return None or some default value if no mode exists
    elif len(modes) > 1:  # More than one mode indicates a tie
        return "TIE"
    else:
        return modes.iloc[0]  # Use .iloc for safe access by index position


def _check_unanimity(row):
    """
    Checks if all answers are the same.
    """
    if row["chatgpt_answer_0"] == row["chatgpt_answer_1"] == row["chatgpt_answer_2"]:
        return row["chatgpt_answer_0"]  # or any as all are the same
    else:
        return "NOT_UNANIMOUS"


def _parse_inference_results_df(df):
    """
    Parses inference results df by computing the average per question type,
    chatgpt vs human.
    """
    df["majority_answer_or_tie"] = df[
        ["chatgpt_answer_0", "chatgpt_answer_1", "chatgpt_answer_2"]
    ].apply(_determine_majority_or_tie, axis=1)
    df["unanimous_or_not"] = df.apply(_check_unanimity, axis=1)

    df["chatgpt_majority_correct"] = df["majority_answer_or_tie"] == df["actual_answer"]
    df["chatgpt_unanimous_correct"] = df["unanimous_or_not"] == df["actual_answer"]

    df["chatgpt_attempt0_correct"] = df["chatgpt_answer_0"] == df["actual_answer"]
    df["chatgpt_attempt1_correct"] = df["chatgpt_answer_1"] == df["actual_answer"]
    df["chatgpt_attempt2_correct"] = df["chatgpt_answer_2"] == df["actual_answer"]

    df["human_correct_percentage"] /= 100

    # Slice by question type.
    df = (
        df.groupby("question_type")[
            [
                "human_correct_percentage",
                "chatgpt_attempt0_correct",
                "chatgpt_attempt1_correct",
                "chatgpt_attempt2_correct",
            ]
        ].mean()
        * 100
    )
    df.reset_index(inplace=True)

    # average the 3 attempts
    df["chatgpt_average_correct_percentage"] = df[
        [
            "chatgpt_attempt0_correct",
            "chatgpt_attempt1_correct",
            "chatgpt_attempt2_correct",
        ]
    ].mean(axis=1)

    df = df[
        [
            "question_type",
            "human_correct_percentage",
            "chatgpt_average_correct_percentage",
        ]
    ]

    return df


def _compute_averages():
    dfs = []
    for exp in list(Experiment):
        if exp == Experiment.HUMAN_CONTROL:
            continue
        filepath = get_result_csvpath_for_experiment(exp)
        df = pd.read_csv(filepath)
        # If you only ran it one time.
        # if exp == Experiment.ZERO_SHOT_GPT4O:
        df["chatgpt_answer_1"] = df["chatgpt_answer_0"]
        df["chatgpt_answer_2"] = df["chatgpt_answer_0"]
        df = _parse_inference_results_df(df)
        df["experiment_name"] = exp.value
        dfs.append(df)

    joined = pd.concat(dfs, ignore_index=True)

    return joined


def _compute_averages_per_attempt_slice_by_question_type():
    results = []
    for exp in list(Experiment):
        filepath = get_result_csvpath_for_experiment(exp)
        df = pd.read_csv(filepath)

        df["chatgpt_attempt0_correct"] = df["chatgpt_answer_0"] == df["actual_answer"]
        df["chatgpt_attempt1_correct"] = df["chatgpt_answer_1"] == df["actual_answer"]
        df["chatgpt_attempt2_correct"] = df["chatgpt_answer_2"] == df["actual_answer"]

        question_type_counts = df["question_type"].value_counts()
        df["num_questions"] = df["question_type"].map(question_type_counts) / 100
        df = (
            df.groupby("question_type")[
                [
                    "chatgpt_attempt0_correct",
                    "chatgpt_attempt1_correct",
                    "chatgpt_attempt2_correct",
                    "num_questions",
                ]
            ].mean()
            * 100
        )
        df.reset_index(inplace=True)
        df["num_questions"] = df["num_questions"].astype(int)
        df = df.replace("ContentType.TEXT_ONLY", "Text only")
        df = df.replace("ContentType.TEXT_AND_IMAGES", "Image based")

        exp_name = exp.value
        for _, row in df.iterrows():
            question_type = row["question_type"]
            num_questions = row["num_questions"]
            for attempt in [0, 1, 2]:
                col = f"chatgpt_attempt{attempt}_correct"
                avg = row[col]
                results.append([exp_name, question_type, num_questions, attempt, avg])

    return results


def _compute_averages_per_attempt(slice_by_question_type):
    if slice_by_question_type:
        return _compute_averages_per_attempt_slice_by_question_type()

    results = []
    for exp in list(Experiment):
        filepath = get_result_csvpath_for_experiment(exp)
        df = pd.read_csv(filepath)

        df["chatgpt_attempt0_correct"] = df["chatgpt_answer_0"] == df["actual_answer"]
        df["chatgpt_attempt1_correct"] = df["chatgpt_answer_1"] == df["actual_answer"]
        df["chatgpt_attempt2_correct"] = df["chatgpt_answer_2"] == df["actual_answer"]

        num_questions = len(df)
        df = (
            df[
                [
                    "chatgpt_attempt0_correct",
                    "chatgpt_attempt1_correct",
                    "chatgpt_attempt2_correct",
                ]
            ].mean()
            * 100
        ).to_frame(name="Value")

        attempt0_avg = df.loc["chatgpt_attempt0_correct", "Value"]
        attempt1_avg = df.loc["chatgpt_attempt1_correct", "Value"]
        attempt2_avg = df.loc["chatgpt_attempt2_correct", "Value"]

        exp_name = exp.value
        question_type = "ALL"
        results.append([exp_name, question_type, num_questions, 0, attempt0_avg])
        results.append([exp_name, question_type, num_questions, 1, attempt1_avg])
        results.append([exp_name, question_type, num_questions, 2, attempt2_avg])

    return results


def _write_df_to_csv(df, filename):
    df = df.replace("ContentType.TEXT_ONLY", "Text only")
    df = df.replace("ContentType.TEXT_AND_IMAGES", "Image based")

    output_path = f"{ROOT_DIR}/out/analysis/{filename}"
    df.to_csv(output_path, index=True)
    print(f"wrote results to {output_path}")


def _write_list_to_csv(results, filename):
    output_path = f"{ROOT_DIR}/out/analysis/{filename}"

    with open(output_path, mode="w", newline="") as file:
        writer = csv.writer(file)

        header = ["exp_name", "question_type", "num_questions", "attempt", "average"]
        writer.writerow(header)
        for result in results:
            writer.writerow(result)
    print(f"wrote results to {output_path}")


averages_df = _compute_averages()
_write_df_to_csv(averages_df, "average-stats.csv")

# averages_per_attempt_df = _compute_averages_per_attempt(slice_by_question_type=True)
# _write_list_to_csv(averages_per_attempt_df, "per-attempt-stats-by-question-type.csv")

# averages_per_attempt_df = _compute_averages_per_attempt(slice_by_question_type=False)
# _write_list_to_csv(averages_per_attempt_df, "per-attempt-stats.csv")

# For some reason (likely because of the circular imports), this module gets
# executed twice. So we manually add an exit.
exit()
