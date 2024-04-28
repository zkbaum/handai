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
from prompt_util import create_prompt
from inference_util import (
    Model,
    HandGPTResponse,
    do_chat_completion,
    parse_response_string_answeronly,
    parse_response_answeronly,
    write_inference_csv,
    InferenceResult,
)
from private import ROOT_DIR
from openai import OpenAI
import time

client = OpenAI()

# question_num_to_assistant = {}
# assistants = client.beta.assistants.list()
# for a in assistants:
#     if 'target_question' in a.metadata:
#         question_num = a.metadata['target_question']
#         if question_num in question_num_to_assistant:
#             print('WARNING, UNEXPECTED DUP ASSISTANT')
#             exit()
#         question_num_to_assistant[question_num] = a.id

# print('Got all assistant IDs. Moving on...')


# print(question_num_to_assistant)
# exit()

# print(assistants.metadata)
super_assistant = client.beta.assistants.retrieve(
    # This is the v1 assistant where I combined everything into a few pdfs.
    # "asst_hcPA7VzvVgw5H8xdEdsQliGx"
)
# eval_questions = assistant.metadata['questions_selected'].split(',')
# eval_questions = [int(x) for x in eval_questions]
eval_set = (
    QuestionsBuilder()
    .year(2013)
    .question_content_type(ContentType.TEXT_ONLY)
    .commentary_content_type(ContentType.TEXT_ONLY)
    .build()
)
# eval_set = [x for x in eval_set if str(x.get_question_number()) in question_num_to_assistant]

references_list = read_references_csv(
    f"{ROOT_DIR}/data/references/handai-2013-references/2013-references.csv"
)
# references_list = []


def query_assistant(client, assistant, exam_question: ExamQuestion):
    thread = client.beta.threads.create()

    message = client.beta.threads.messages.create(
        thread_id=thread.id,
        role="user",
        content=f"<question>{exam_question.format_question()}</question>",
    )

    run = client.beta.threads.runs.create(
        thread_id=thread.id,
        assistant_id=assistant.id,
        # instructions = INSTRUCTIONS_NO_EXEMPLARS
        additional_instructions="Please make sure the response is in the form <discussion>insert discussion</discussion> <answer>C</answer>. Even if you are unsure, please pick one letter you are most confident about.",
    )

    while run.status in ["queued", "in_progress", "cancelling"]:
        time.sleep(1)  # Wait for 1 second
        run = client.beta.threads.runs.retrieve(thread_id=thread.id, run_id=run.id)

    if run.status == "completed":
        messages = client.beta.threads.messages.list(thread_id=thread.id)
        return messages
        # messages = [m for m in messages]
        # messages.reverse()
        # for message in messages:
        #     # unclear if we will ever have more than 1 content...TBD
        #     print(f'{message.role}: {message.content}\n' )
        #     if message.content[0].text.annotations:
        #         print(f'Found annotation: {message.content[0].text.annotations}')

        # print(messages)
    else:
        print(f"[ERROR] failed run: {run}")
        return None


def _run_assistant_inference(client, assistant, entry):
    messages = query_assistant(client, assistant, entry)
    retry_count = 0
    # Can't remember why I added this...
    while messages is None and retry_count < MAX_RETRIES:
        print("retrying server...")
        messages = query_assistant(client, assistant, entry)
        retry_count += 1

    txt = messages.data[0].content[0].text.value
    _, chatgpt_answer = parse_response_string_answeronly(txt)

    # Retry up until max retry threshold.
    retry_count = 0
    while chatgpt_answer == "PARSE_ERROR" and retry_count < MAX_RETRIES:
        print(
            "[INFO] failed to find <answer> in response. retrying attempt {}".format(
                retry_count
            )
        )
        messages = query_assistant(client, assistant, entry)
        if messages is None:
            print(
                "[WARNING] uh oh...messages is none. this probably should not happen. ignoring the house down boots :)"
            )
            continue
        txt = messages.data[0].content[0].text.value
        _, chatgpt_answer = parse_response_string_answeronly(txt)
        retry_count += 1
    if chatgpt_answer == "PARSE_ERROR":
        print(
            "[WARNING] failed to parse {} times, this will count as incorrect answer".format(
                MAX_RETRIES + 1
            )
        )

    response = HandGPTResponse(
        raw_response=messages,
        discussion=txt,
        answer=chatgpt_answer,
        citations=messages.data[0].content[0].text.annotations,
    )
    return response


results = []
i = 0
MAX_RETRIES = 3
ENSEMBLING_COUNT = 2


for entry in eval_set:
    question_num = str(entry.get_question_number())
    print(
        "handling question {} of {} (y={}, q={})".format(
            i, len(eval_set), entry.get_year(), question_num
        )
    )

    # assistant_id = question_num_to_assistant[question_num]
    # print(f'  fetching assistant for question {question_num} with id {assistant_id}')
    # assistant = client.beta.assistants.retrieve(assistant_id)
    # print('  fetched the assistant.')

    assistant = super_assistant

    responses = []
    for n in range(ENSEMBLING_COUNT):
        print(f"    doing ensembling query {n} of {ENSEMBLING_COUNT}")
        response = _run_assistant_inference(client, assistant, entry)
        responses.append(response)

    results.append(
        InferenceResult(
            question=entry,
            prompt=None,
            question_type=entry.get_question_content_type(),
            model=Model.GPT4,
            responses=responses,
        )
    )
    i += 1

write_inference_csv(
    results,
    references_list=references_list,
    year=2013,
    exp_name="assistants-v1",
)
