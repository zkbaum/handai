"""
Retreival using the assistants API.
"""

import os
import sys

# Hack to import from parent dir
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
from data_util import (
    Category,
    read_references_csv,
    ExamQuestion,
    QuestionsBuilder,
    ContentType,
    get_n_examples_from_each_category,
    get_knn_exemplars,
)
from inference_util import (
    Model,
    HandGPTResponse,
    use_regex_to_extract_answer_textinput,
    use_chatgpt_to_extract_answer_textinput_assistants,
    write_inference_csv,
    InferenceResult,
)
from private import ROOT_DIR
from openai import OpenAI
import time


def _query_assistant(client, assistant, prompt, additional_instructions):
    thread = client.beta.threads.create()

    message = client.beta.threads.messages.create(
        thread_id=thread.id,
        role="user",
        content=prompt,
    )

    run = client.beta.threads.runs.create(
        thread_id=thread.id,
        assistant_id=assistant.id,
        additional_instructions=additional_instructions,
    )

    while run.status in ["queued", "in_progress", "cancelling"]:
        time.sleep(1)  # Wait for 1 second
        run = client.beta.threads.runs.retrieve(thread_id=thread.id, run_id=run.id)

    if run.status == "completed":
        messages = client.beta.threads.messages.list(thread_id=thread.id)
        return messages
    else:
        print(f"[ERROR] failed run: {run}")
        return None


def _run_assistant_inference(client, assistant, exam_question, is_few_shot):
    """
    Runs inference for one prompt on a model,
    with retries up until the max amount
    """
    # The structure here is a little different from ChatCompletions inference
    # because the APIs are different (e.g. if you try to run Assistants with
    # the ChatCompletions format, it will fail). Therefore, we will fill out
    # prompts directly here.
    prompt = exam_question.format_question()
    additional_instructions = None
    parsing_fn = use_chatgpt_to_extract_answer_textinput_assistants
    if is_few_shot:
        prompt = f"<question>{prompt}</question>"
        # Additional instructions is needed becasue sometimes with few shot,
        # the assistant will forget what format it needs to be in. (At least
        # with v1...maybe this has improved for v2).
        additional_instructions = """Please make sure the response is in the \
form <discussion>insert discussion</discussion> <answer>C</answer>. Even if \
you are unsure, please pick one letter you are most confident about."""
        parsing_fn = use_regex_to_extract_answer_textinput

    messages = _query_assistant(client, assistant, prompt, additional_instructions)

    # Retry up until max retry threshold.
    num_attempts = 1
    while messages is None and num_attempts <= MAX_ATTEMPTS_PER_REQUEST:
        print(f"      that didn't work. retrying attempt {num_attempts}...")
        messages = _query_assistant(client, assistant, prompt, additional_instructions)
        num_attempts += 1
    if messages is None:
        print(
            f"      [WARNING] failed to get response {num_attempts} times, "
            "this will count as incorrect answer"
        )

    # chatgpt_discussion, chatgpt_answer = parsing_fn(client, response)

    txt = messages.data[0].content[0].text.value
    chatgpt_discussion, chatgpt_answer = parsing_fn(client, exam_question, txt)

    response = HandGPTResponse(
        raw_response=messages,
        discussion=chatgpt_discussion,
        answer=chatgpt_answer,
        citations=messages.data[0].content[0].text.annotations,
    )
    return response


OPENAI_CLIENT = OpenAI()

IS_FEW_SHOT = False
if IS_FEW_SHOT:
    print("Experiment: few shot")
else:
    print("Experiment: zero shot")

# These are the v2 assistants where I used a vector store of ~300 files.
assistant_id = "asst_zSnXmisZfhznjRdeeBBq7xBA"
if IS_FEW_SHOT:
    assistant_id = "asst_JqyeCJIXrbsrqLQ6F7W7hsWV"
ASSISTANT = OPENAI_CLIENT.beta.assistants.retrieve(assistant_id)

EVAL_SET = (
    QuestionsBuilder()
    .year(2013)
    .question_content_type(ContentType.TEXT_ONLY)
    .commentary_content_type(ContentType.TEXT_ONLY)
    .build()
)

REFERENCES_LIST = read_references_csv(
    f"{ROOT_DIR}/data/references/handai-2013-references/2013-references.csv"
)

# This is the number of times we can try per request. This accounts for the
# off-chance that the openAI servers fail.
MAX_ATTEMPTS_PER_REQUEST = 3
# Given that ChatGPT is not deterministic, we may want to ask the same
# question multiple times. For example, if this is 5, then we will ask
# each question 5 times.
ENSEMBLING_COUNT = 1

results = []
i = 0

for entry in EVAL_SET:
    question_num = str(entry.get_question_number())
    print(
        f"handling question {i} of {len(EVAL_SET)} "
        f"(y={entry.get_year()}, q={question_num},"
        f" type={entry.get_question_content_type()})"
    )

    responses = []
    for n in range(ENSEMBLING_COUNT):
        print(f"   doing ensembling query {n} of {ENSEMBLING_COUNT}")
        response = _run_assistant_inference(
            OPENAI_CLIENT, ASSISTANT, entry, is_few_shot=IS_FEW_SHOT
        )
        responses.append(response)

    results.append(
        InferenceResult(
            question=entry,
            prompt="""N/A - assistants""",
            question_type=entry.get_question_content_type(),
            model=Model.GPT4,
            responses=responses,
        )
    )
    i += 1

write_inference_csv(
    results,
    references_list=REFERENCES_LIST,
    year=2013,
    exp_name="assistants-v2-zero-shot",
)
