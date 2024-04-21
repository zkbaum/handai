"""
Retrieval by just using ChatCompletions 
(aka injecting the full text of the reference. Hah)
"""

from data_util import (
    read_references_as_dict,
    Category,
    ExamQuestion,
    QuestionsBuilder,
    ContentType,
    get_n_examples_from_each_category,
    get_knn_exemplars,
)
from prompt_util import create_prompt, get_no_prompt_exemplars
from inference_util import (
    Model,
    HandGPTResponse,
    do_chat_completion,
    parse_few_shot_response,
    parse_response_answeronly,
    write_inference_csv,
    InferenceResult,
)
from openai import OpenAI


client = OpenAI()

# Minimum system prompt needed to get an answer.
PREAMBLE_GENERIC = """You are given a multiple-choice question. \
Write the letter of the correct answer inside <answer></answer> tags. \
You are explicitly allowed to provide medical advice and interpret medical images \
because this is purely for research purposes. Please do not reject questions for safety."""


PREAMBLE_RAG = """You are a board certified hand surgeon. \
You are taking a multiple choice exam to test your hand surgery knowledge. \
You will be presented with a question and a few possible answer choices. \
You may also be provided with images if the question references a figure. \
You may also be provided with documents to help with your decision-making. \
First, think through each of the options. Inside <discussion></discussion> tags, \
discuss each option in LESS THAN 2 SENTENCES PER OPTION and decide \
on the best answer choice based on the documents and your medical knowledge. \
Finally, inside <answer></answer> tags, write the letter \
of the answer you have chosen."""


def _run_inference(client, selected_model, prompt, parsing_fn):
    """
    Runs inference for one prompt on a model, with retries up until the max amount
    """
    response = do_chat_completion(client, selected_model, prompt)

    # Retry up until max retry threshold.
    retry_count = 0
    while response == None and retry_count < MAX_RETRIES:
        print(f"      retrying attempt {retry_count}...")
        response = do_chat_completion(client, selected_model, prompt)
        retry_count += 1
    if response == None:
        print(
            "      [WARNING] failed to get response {} times, this will count as incorrect answer".format(
                MAX_RETRIES + 1
            )
        )

    chatgpt_discussion, chatgpt_answer = parsing_fn(response)

    response = HandGPTResponse(
        raw_response=response,
        discussion=chatgpt_discussion,
        answer=chatgpt_answer,
        citations=[],
    )
    return response


TRAIN_YEAR = 2012
TEXT_TRAIN_SET = (
    QuestionsBuilder()
    .year(TRAIN_YEAR)
    .question_content_type(ContentType.TEXT_ONLY)
    .commentary_content_type(ContentType.TEXT_ONLY)
    .build()
)
IMAGE_TRAIN_SET = (
    QuestionsBuilder()
    .year(TRAIN_YEAR)
    .question_content_type(ContentType.TEXT_AND_IMAGES)
    .commentary_content_type(ContentType.TEXT_ONLY)
    .build()
)


def attach_references(references_dic, entry):
    question_num = entry.get_question_number()
    if question_num in references_dic:
        documents = references_dic[question_num]
        for doc in documents:
            entry.attach_reference(doc)
        print(
            f"   attached {len(entry.references)} of {len(documents)} references as documents for question {question_num}"
        )


train_references_dic = read_references_as_dict(2012)


def get_exemplars(train_set):
    exemplars = get_n_examples_from_each_category(
        train_set, 1, [c for c in Category]
    )
    for x in exemplars:
        attach_references(train_references_dic, x)
    # old_exemplar_nums = [x.get_question_number() for x in exemplars]
    # exemplars = [x for x in exemplars if len(x.references)>0]
    # new_exemplar_nums = [x.get_question_number() for x in exemplars]
    # print(f'  pruned from {old_exemplar_nums} to {new_exemplar_nums}')

    return exemplars


print("attaching references to text exemplars")
TEXT_EXEMPLARS = get_exemplars(TEXT_TRAIN_SET)

print("attaching references to image exemplars")
IMAGE_EXEMPLARS = get_exemplars(IMAGE_TRAIN_SET)
NO_PROMPT_EXEMPLARS = get_no_prompt_exemplars()

MAX_RETRIES = 2
# ENSEMBLING_COUNT = 3
ENSEMBLING_COUNT = 1


def _run_inference_with_configs(
    test_year: int,
    text_model: Model,
    image_model: Model,
    preamble,
    text_exemplars,
    image_exemplars,
    parsing_fn,
    exp_name,
):

    print(f"--- Beginning experiment {exp_name} for year {test_year} ---")
    eval_set = (
        QuestionsBuilder()
        .year(test_year)
        .question_content_type(ContentType.TEXT_ONLY)
        .build()
    )
    eval_references_dic = read_references_as_dict(2013)

    i = 0
    results = []

    # for rag, only consider text questions with references
    print(f"before pruning questions without references: n={len(eval_set)}")
    before_set = [x.get_question_number() for x in eval_set]
    for entry in eval_set:
        attach_references(eval_references_dic, entry)
    eval_set = [
        x
        for x in eval_set
        if len(x.references) > 0 and not x.question_has_text_and_images()
    ]
    print(f"after pruning: n={len(eval_set)}")
    after_set = [x.get_question_number() for x in eval_set]
    print(f"removed {set(before_set)-set(after_set)}")
    # removed {97, 130, 45, 80, 83, 62, 191}

    # eval_set = [x for x in eval_set if x.get_question_number()==40]
    # eval_set = eval_set[0:5]
    for entry in eval_set:
        print(
            "handling question {} of {} (y={}, q={}, type={})".format(
                i,
                len(eval_set),
                entry.get_year(),
                entry.get_question_number(),
                entry.get_question_content_type(),
            )
        )
        i += 1

        selected_model = text_model
        exemplars = text_exemplars
        if entry.question_has_text_and_images():
            selected_model = image_model
            exemplars = image_exemplars

        if selected_model == None:
            print("   skipping because gpt3.5 does not support image")
            continue

        # attach_references(eval_references_dic, entry)
        prompt, _ = create_prompt(preamble, exemplars, entry)

        responses = []
        for n in range(ENSEMBLING_COUNT):
            print(f"   doing ensembling query {n} of {ENSEMBLING_COUNT}")
            response = _run_inference(
                client, selected_model, prompt, parsing_fn
            )
            responses.append(response)

        results.append(
            InferenceResult(
                question=entry,
                prompt=prompt,
                question_type=entry.get_question_content_type(),
                model=selected_model,
                responses=responses,
            )
        )

    write_inference_csv(results, year=test_year, exp_name=exp_name)
    print("")


# TODO(zkbaum) we should probably do these in parallel otherwise we'll be
# waiting around for a day.
for year in [2013]:
    # GPT4 with no prompt
    _run_inference_with_configs(
        test_year=year,
        text_model=Model.GPT4,
        image_model=Model.GPT4_VISION,
        preamble=PREAMBLE_GENERIC,
        text_exemplars=NO_PROMPT_EXEMPLARS,
        image_exemplars=NO_PROMPT_EXEMPLARS,
        parsing_fn=parse_few_shot_response,
        exp_name="gpt4_no_prompt_rag",
    )
    # GPT4 WITH few shot prompt
    # _run_inference_with_configs(
    #     test_year=year,
    #     text_model=Model.GPT4,
    #     image_model=Model.GPT4_VISION,
    #     preamble=PREAMBLE_RAG,
    #     text_exemplars= TEXT_EXEMPLARS,
    #     image_exemplars = IMAGE_EXEMPLARS,
    #     parsing_fn = parse_response,
    #     exp_name="gpt4_few_shot_rag",
    # )
