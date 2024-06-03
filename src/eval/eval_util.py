import pandas as pd
from enum import Enum
import os
import sys

# Hack to import from parent dir
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
from private import get_result_csvpath_for_experiment, get_key_path, get_human_path


class Experiment(Enum):
    HUMAN_CONTROL = "human"
    GPT3_5 = "gpt3.5"
    GPT4 = "gpt4"
    GPT4O = "gpt4o"
    GPT4O_BETTER_PROMPT = "gpt4o-better-prompt"
    GPT4O_FILE_SEARCH = "gpt4-file-search"
    GPT4O_FILE_SEARCH_AND_BETTER_PROMPT = "gpt4-file-search-and-better-prompt"


def get_key_df():
    # CSV in the form [question_id, correct_answer, question_type]
    return pd.read_csv(get_key_path())


def get_human_df():
    # CSV in the form [student_id, answer1, answer2, ..., answer200]
    return pd.read_csv(get_human_path())


def get_chatgpt_df(year: int, experiment: Experiment):
    # # CSV in the form [question_id, question_type,actual_answer, human_correct_percentage, attempt0, attempt1, attempt2, .., attemptn]
    # gpt_3_5_df = pd.read_csv(f"{ROOT_DIR}/gpt3-5.csv")
    filepath = get_result_csvpath_for_experiment(year, experiment)
    df = pd.read_csv(filepath)

    for i in range(10):
        df.rename(columns={f"chatgpt_answer_{i}": f"attempt{i}"}, inplace=True)

    selected_indices = [
        "question_number",
        "question_type",
        "actual_answer",
        "human_correct_percentage",
    ] + [f"attempt{i}" for i in range(10)]
    df = df[selected_indices]
    df = df.replace("ContentType.TEXT_ONLY", "Text")
    df = df.replace("ContentType.TEXT_AND_IMAGES", "Image")
    df["question_number"] = df["question_number"].apply(lambda x: f"question{x}")
    df.rename(columns={"question_number": "question_id"}, inplace=True)

    return df


# print(_get_chatgpt_df(Experiment.ZERO_SHOT_GPT3_5))
