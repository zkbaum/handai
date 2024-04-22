"""
Example of creating finetuning splits. Doesn't actually work because the APIs
have changed...but should give you an idea if you ever have to get back to it
later.
"""

from data_util import Exam, get_exam_data, get_n_examples_from_each_category
from prompt_util import generate_prompt_messages
from inference_util import (
    Model,
    do_chat_completion,
    use_regex_to_extract_answer,
    write_inference_csv,
)
import random
import json
from private import ROOT_DIR

questions_2011 = get_exam_data(Exam.YEAR_2011, text_only=True)
questions_2012 = get_exam_data(Exam.YEAR_2012, text_only=True)


# Create a train/validation split. Since there are no categories for 2011, we must do it randomly.
random.shuffle(questions_2011)
TRAIN_VALIDATION_SPLIT = 0.8
split_index = int(TRAIN_VALIDATION_SPLIT * len(questions_2011))
train_set = questions_2011[:split_index]
validation_set = questions_2011[split_index:]

print("input dataset is size:  {}".format(len(questions_2011)))
print("  using split ratio: {}".format(TRAIN_VALIDATION_SPLIT))
print("  created train set of size: {}".format(len(train_set)))
print("  created validation set of size: {}".format(len(validation_set)))

PREAMBLE = """You are a board certified hand surgeon. \
You are taking a multiple choice exam to test your hand surgery knowledge. \
You will be presented with a question and a few possible answer choices. \
First, think through each of the options. Inside <discussion></discussion> tags, \
discuss each option in LESS THAN 2 SENTANCES PER OPTION and decide \
on the best answer choice. Then, inside <answer></answer> tags, write the letter \
of the answer you have chosen."""

# Get exemplars. To ensure we are using different ones from the inference, we will reverse the list :)
questions_2012.reverse()
exemplars = get_n_examples_from_each_category(questions_2012, 2)


def write_jsonl(name, dataset):
    """
    Writes output jsonl file. Assumes output files should live in
    $ROOT_DIR/out/finetune_splits/
    """
    filepath = f"{ROOT_DIR}/out/finetune_splits/{name}.jsonl"
    with open(filepath, "w") as file:
        for entry in dataset:
            # For now we will not use exemplars with fine-tuning...
            prompt, target = generate_prompt_messages(
                PREAMBLE, exemplars, entry
            )
            json_record = json.dumps({"messages": prompt + target})
            file.write(json_record + "\n")

    print(f"JSONL file '{filepath}' created successfully.")


write_jsonl("2011-textonly-with-exemplars-train", train_set)
write_jsonl("2011-textonly-with-exemplars-validation", validation_set)
