"""
Script to analyze results from a hand AI experiment.
"""

import pandas as pd
from enum import Enum
from private import get_result_csvpath_for_experiment, ROOT_DIR


class Experiment(Enum):
    ZERO_SHOT_GPT3_5 = "zero-shot-gpt3.5"
    ZERO_SHOT_GPT4 = "zero-shot-gpt4"
    FEW_SHOT_GPT4 = "few-shot-gpt4"
    RAG_ZERO_SHOT = "rag-zero-shot"
    RAG_FEW_SHOT = "rag-few-shot"


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

    # Average the 3 attempts.
    df["chatgpt_average_correct_percentage"] = df[
        [
            "chatgpt_attempt0_correct",
            "chatgpt_attempt1_correct",
            "chatgpt_attempt2_correct",
        ]
    ].mean(axis=1)

    return df[
        [
            "question_type",
            "human_correct_percentage",
            "chatgpt_average_correct_percentage",
        ]
    ]


dfs = []
for exp in list(Experiment):
    filepath = get_result_csvpath_for_experiment(exp)
    df = pd.read_csv(filepath)
    # If you only ran it one time.
    # if exp == Experiment.RAG_ZERO_SHOT:
    #     df["chatgpt_answer_1"] = df["chatgpt_answer_0"]
    #     df["chatgpt_answer_2"] = df["chatgpt_answer_0"]
    df = _parse_inference_results_df(df)
    df["experiment_name"] = exp.value
    dfs.append(df)

joined = pd.concat(dfs, ignore_index=True)
joined = joined.replace("ContentType.TEXT_ONLY", "Text only")
joined = joined.replace("ContentType.TEXT_AND_IMAGES", "Image based")

output_path = f"{ROOT_DIR}/out/analysis/stats.csv"
joined.to_csv(output_path, index=True)
print(f"wrote results to {output_path}")

# For some reason (likely because of the circular imports), this module gets
# executed twice. So we manually add an exit.
exit()
