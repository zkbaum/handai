"""
This is the main entry point for inference. It supports zero and few-shot 
inference.
"""

from openai import OpenAI

from data_util import (
    Category,
    QuestionsBuilder,
    ContentType,
    get_n_examples_from_each_category,
)
from prompt_util import create_prompt
from inference_util import (
    Model,
    HandGPTResponse,
    do_chat_completion,
    use_regex_to_extract_answer_chatcompletion,
    write_inference_csv,
    InferenceResult,
    use_chatgpt_to_extract_answer,
)

CLIENT = OpenAI()

PREAMBLE_DETAILED = """You are a board certified hand surgeon. \
You are taking a multiple choice exam to test your hand surgery knowledge. \
You will be presented with a question and a few possible answer choices. \
You may also be provided with images if the question references a figure. \
First, think through each of the options. Inside <discussion></discussion> \
tags, briefly discuss each option and decide on the best answer choice. Then, \
inside <answer></answer> tags, write the letter of the answer you have chosen.
"""

# This is the number of times we can try per request. This accounts for the
# off-chance that the openAI servers fail.
MAX_ATTEMPTS_PER_REQUEST = 3
# Given that ChatGPT is not deterministic, we may want to ask the same
# question multiple times. For example, if this is 5, then we will ask
# each question 5 times.
ENSEMBLING_COUNT = 10


def _run_inference(client, entry, selected_model, prompt, parsing_fn):
    """
    Runs inference for one prompt on a model,
    with retries up until the max amount
    """
    response = do_chat_completion(client, selected_model, prompt)

    # Retry up until max retry threshold.
    num_attempts = 1
    while response is None and num_attempts <= MAX_ATTEMPTS_PER_REQUEST:
        print(f"      that didn't work. retrying attempt {num_attempts}...")
        response = do_chat_completion(client, selected_model, prompt)
        num_attempts += 1
    if response is None:
        print(
            f"      [WARNING] failed to get response {num_attempts} times, "
            "this will count as incorrect answer"
        )

    chatgpt_discussion, chatgpt_answer = parsing_fn(client, selected_model, entry, response)        

    response = HandGPTResponse(
        raw_response=response,
        discussion=chatgpt_discussion,
        answer=chatgpt_answer,
        citations=[],
    )
    return response


TRAIN_YEAR = 2008
TEXT_TRAIN_SET = (
    QuestionsBuilder()
    .year(TRAIN_YEAR)
    .question_content_type(ContentType.TEXT_ONLY)
    .commentary_content_type(ContentType.TEXT_ONLY)
    .build()
)

# We are deliberately only using text exemplars because quality seems to
# decrease when we use image-based questions in exemplars.
TEXT_EXEMPLARS = get_n_examples_from_each_category(TEXT_TRAIN_SET, 1, list(Category))


def _run_inference_with_configs(
    test_year: int,
    model: Model,
    preamble,
    exemplars,
    parsing_fn,
    exp_name,
) -> str:
    """
    Returns:
        results output file string
    """

    print(f"--- Beginning experiment {exp_name} for year {test_year} ---")
    eval_set = QuestionsBuilder().year(test_year).build()

    i = 0
    results = []
    for entry in eval_set:
        print(
            f"handling question {i} of {len(eval_set)} "
            f"(y={entry.get_year()}, q={entry.get_question_number()},"
            f" type={entry.get_question_content_type()})"
        )
        i += 1

        if entry.question_has_text_and_images() and model == Model.GPT3_5:
            print("   skipping because gpt3.5 does not support image")
            continue

       
        prompt, _ = create_prompt(preamble, exemplars, entry)

        responses = []
        for n in range(ENSEMBLING_COUNT):
            print(f" doing ensembling query {n} of {ENSEMBLING_COUNT}")
            response = _run_inference(CLIENT, entry, model, prompt, parsing_fn)
            if response.answer == 'EXTRACTION_ERROR_RATELIMIT':
                print("[GRACEFUL EXIT WARNING] Hit quota limit so ending gracefully")
                return write_inference_csv(results, year=test_year, exp_name=exp_name)
            # print(f"[debug] got response: {response}")
            responses.append(response)

        results.append(
            InferenceResult(
                question=entry,
                prompt=prompt,
                question_type=entry.get_question_content_type(),
                model=model,
                responses=responses,
            )
        )

    result_filepath = write_inference_csv(results, year=test_year, exp_name=exp_name)
    print("")
    return result_filepath


# TODO(zkbaum) we should probably do these in parallel otherwise we'll be
# waiting around for a day.
# for year in [2009, 2010, 2011, 2012, 2013]:
paths = []
for year in [2013]:
    # GPT3.5 zero-shot
    paths.append(
        _run_inference_with_configs(
            test_year=year,
            model=Model.GPT3_5,
            preamble=None,
            exemplars=None,
            parsing_fn=use_chatgpt_to_extract_answer,
            exp_name="gpt3_zero_shot",
        )
    )
    # GPT4 zero-shot
    # paths.append(
    #     _run_inference_with_configs(
    #         test_year=year,
    #         model=Model.GPT4,
    #         preamble=None,
    #         exemplars=None,
    #         parsing_fn=use_chatgpt_to_extract_answer,
    #         exp_name="gpt4_zero_shot_shard4",
    #     )
    # )
    # GPT4o zero-shot
    paths.append(
        _run_inference_with_configs(
            test_year=year,
            model=Model.GPT4O,
            preamble=None,
            exemplars=None,
            parsing_fn=use_chatgpt_to_extract_answer,
            exp_name="gpt4o_zero_shot",
        )
    )
    # # GPT4 few shot
    paths.append(
        _run_inference_with_configs(
            test_year=year,
            model=Model.GPT4O,
            preamble=PREAMBLE_DETAILED,
            exemplars=TEXT_EXEMPLARS,
            parsing_fn=use_regex_to_extract_answer_chatcompletion,
            exp_name="gpt4o_few_shot",
        )
    )

print(f"See output at following paths:\n{"\n".join(paths)}")
print("done :)")
