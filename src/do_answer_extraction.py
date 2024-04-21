"""
As part of few-shot prompting experiments, we have a seperate model to extract
the single letter answer from responses. Usuallyt we run this as part of 
run_inference. However, if we already have the inferences, we can run this
one-off script to compute everything.
"""

from openai import OpenAI

import pandas as pd
from inference_util import _use_chatgpt_to_extract_answer_internal

CLIENT = OpenAI()


def _read_inference_csv_and_extract_answers(client, filepath):
    df = pd.read_csv(filepath)
    for index, row in df.iterrows():
        print(f'handling question {row["question_number"]}')
        # in place update the answer :)
        _, extracted = _use_chatgpt_to_extract_answer_internal(
            client, row["chatgpt_discussion_0"]
        )
        df.loc[index, "chatgpt_answer_0"] = extracted

    return df


INPUT_PATH = "/PATH/TO/INPUT"
OUTPUT_PATH = "/PATH/TO/OUTPUT"

df = _read_inference_csv_and_extract_answers(CLIENT, INPUT_PATH)
df.to_csv(OUTPUT_PATH, index=True)
print(f"wrote results to {OUTPUT_PATH}")
