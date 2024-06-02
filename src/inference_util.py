"""
Helper functions for doing inference, parsing output, and writing to csv.
"""

from enum import Enum, auto
import re
from datetime import datetime
from data_util import ExamQuestion, Reference, ContentType
from dataclasses import dataclass
import csv
from private import (
    ROOT_DIR,
    EXEMPLARS_FOR_GPT3_EXTRACTOR,
    EXEMPLARS_FOR_GPT4_EXTRACTOR,
    ASSISTANTS_EXEMPLARS_FOR_EXTRACTOR,
    get_file_id_to_reference_mappings_2013,
)
from openai import RateLimitError


# A few notes about models
# GPT4 can fit all of 2012 (exam has 35K tokens, context window 128K)
# GPT3.5 curr can fit about 2 questions from each category (context window 4096)
# gpt-3.5-turbo-0125 (new) can fit about 9 questions from each category (context window 16)
class Model(Enum):
    GPT3_5 = auto()
    GPT4 = auto()
    FtFeb11NoExemplars = auto()
    FtFeb11WithExamplars = auto()
    GPT4_VISION = auto()
    GPT4O = auto()


def get_model_string(model: Model):
    model_enum_to_str = {
        Model.GPT3_5: "gpt-3.5-turbo-0125",
        Model.GPT4: "gpt-4-turbo-2024-04-09",
        Model.GPT4O: "gpt-4o",
        # GPT4_VISION is no longer needed now that gpt-4-turbo support vision.
        # Model.GPT4_VISION: "gpt-4-vision-preview",
        Model.FtFeb11NoExemplars: "ft:gpt-3.5-turbo-1106:personal::8qxFN6cX",
        Model.FtFeb11WithExamplars: "ft:gpt-3.5-turbo-1106:personal::8qxNawaE",
    }
    return model_enum_to_str[model]


@dataclass
class HandGPTResponse:
    raw_response: any
    discussion: str
    answer: str
    citations: any


@dataclass
class InferenceResult:
    question: ExamQuestion
    prompt: any
    model: Model
    question_type: ContentType
    responses: "list[HandGPTResponse]"


def do_chat_completion(client, model: Model, prompt):
    try:
        response = client.chat.completions.create(
            model=get_model_string(model),
            messages=prompt,
            # We will only use the default knobs because that's
            # how real people will experience it.
            # temperature=TEMPERATURE,
            # max_tokens=MAX_TOKENS,
            # top_p=TOP_P,
            # frequency_penalty=FREQUENCY_PENALTY,
            # presence_penalty=PRESENCE_PENALTY
            # response_format={ "type": "json_object" },
        )
        return response
    # Sometimes the servers fail for whatever reason, so let's catch that.
    except Exception as e:
        print(f"[ERROR] Got error with inference: {e}")
        # print("[INFO] prompt used: {}".format(prompt))
        return None


def parse_response_string_answeronly(txt: str):
    pattern = r"<answer>(.*?)<\/answer>"

    # Searching for the pattern in the txt
    match = re.search(pattern, txt, re.DOTALL)

    # Extracting and printing the content if found
    if match:
        answer_content = match.group(1)
        return "N/A", answer_content
    else:
        # print("[INFO] No content found within <discussion> or <answer> tags.")

        return "PARSE_ERROR", "PARSE_ERROR"


def parse_response_answeronly(response):
    if response is None:
        return "PARSE_ERROR", "PARSE_ERROR"

    txt = response.choices[0].message.content
    return parse_response_string_answeronly(txt)


def parse_response_string(txt: str):
    pattern = r"<discussion>(.*?)<\/discussion>\s*<answer>(.*?)<\/answer>"

    # Searching for the pattern in the txt
    match = re.search(pattern, txt, re.DOTALL)

    # Extracting and printing the content if found
    if match:
        discussion_content = match.group(1)
        answer_content = match.group(2)
        return discussion_content, answer_content
    else:
        # print("[INFO] No content found within <discussion> or <answer> tags.")

        return "PARSE_ERROR", "PARSE_ERROR"


def use_regex_to_extract_answer_chatcompletion(client, model, exam_question, response):
    """
    Parses the few-shot response using regex.
    Note 'client' and `model` are unused...this is a hack because the zero shot
    requires these and I wanted the abstraction to work :)
    """
    # TODO(ZKBAUM) clean this up. this happens when there was a fatal
    # inference error, so you should plumb that instead
    if response is None:
        return "PARSE_ERROR", "PARSE_ERROR"

    txt = response.choices[0].message.content

    discussion, answer = parse_response_string(txt)
    print(f"   used regex to extract answer {answer}")
    return discussion, answer


def use_regex_to_extract_answer_assistants(client, exam_question, txt):
    """
    Parses the few-shot response using regex.
    Note 'client' is unused...this is a hack because the zero shot
    requires client and I wanted the abstraction to work :)
    """

    discussion, answer = parse_response_string(txt)
    print(f"   used regex to extract answer {answer}")
    return discussion, answer


def _replace_citations(raw_discussion, citations, file_id_mapping):
    # Hard coding the really big files...
    # Note we assume
    if not file_id_mapping:
        file_id_mapping = get_file_id_to_reference_mappings_2013()

    final_discussion = raw_discussion
    num = 1
    # replace the citations (fileid-abcd) with the actual citation (Kirschenbaum D, etc)
    for citation in citations:
        ref = file_id_mapping[citation.file_citation.file_id]
        quote = citation.file_citation.quote
        final_discussion += f"\n\n[{num}] {ref.reference} ({ref.url})"
        if quote:
            final_discussion += f"\nQuote: {quote}"
        final_discussion = final_discussion.replace(citation.text, f"[{num}]")
        num += 1
    return final_discussion


# TODO(zkbaum) fix replacement for references
def write_inference_csv(
    results: "list[InferenceResult]",
    references_list: "list[Reference]" = [],
    year: int = 0,
    exp_name: str = "",
) -> str:
    """
    Writes inference results to $ROOT_DIR/out/inference

    Returns:
        results file written to
    """
    current_timestamp = datetime.now()
    formatted_timestamp = current_timestamp.strftime("%Y%m%d_%H:%M:%S")
    filepath = f"{ROOT_DIR}/out/inference/{year}_{exp_name}_{formatted_timestamp}.csv"

    # print(references_list)
    # build using list
    file_id_mapping = {}
    for ref in references_list:
        # skip non-ids
        if not ref.is_uploaded:
            continue
        if ref.openai_file_id in file_id_mapping:
            print("WARNING: FATAL ERROR. unexpected...should only be one")
        file_id_mapping[ref.openai_file_id] = ref

    # Open the file in write mode ('w') and create a csv.writer object
    with open(filepath, "w", newline="") as file:
        writer = csv.writer(file)
        header = [
            "question_year",
            "question_number",
            "question_id",
            "category",
            "question",
            "prompt",
            "model",
            "question_type",
            "responses",
        ]
        ensembling_count = len(results[0].responses)
        for n in range(ensembling_count):
            header += [f"chatgpt_discussion_{n}", f"chatgpt_answer_{n}"]
        header += [
            "actual_discussion",
            "actual_answer",
            "human_correct_percentage",
            "human_distribution",
        ]
        writer.writerow(header)

        # Write the data to the CSV file
        for result in results:
            row = [
                result.question.get_year(),
                result.question.get_question_number(),
                result.question.question_id,
                result.question.category,
                result.question.format_question(),
                # unclear why this stops showing up in google doc...perhaps it won't show above a certain limit
                result.prompt,
                result.model,
                result.question_type,
                result.responses,
            ]
            for response in result.responses:
                # citations look like 【43:1†question_1_reference_2.pdf】.
                # this is not very readable, so let's replace them with a format that's easier to understand.
                discussion_with_readable_citations = _replace_citations(
                    response.discussion, response.citations, file_id_mapping
                )
                row += [discussion_with_readable_citations, response.answer]
            row += [
                result.question.get_clean_commentary(),
                result.question.get_correct_answer(),
                result.question.correct_answer_percentage,
                result.question.get_human_distribution(),
            ]
            writer.writerow(row)

    print(f"Data has been written to {filepath}")
    return filepath


def use_chatgpt_to_extract_answer(
    client,
    model,
    exam_question: ExamQuestion,
    original_response,
):
    """
    In the zero-shot approach, ChatGPT gives inconsistent output formatting.
    Therefore, we will use a seperate model to extract the answer.
    """
    print("   got zero-shot response, extracting answer...")
    if original_response:
        original_response = original_response.choices[0].message.content

    extractor_prompt = [
        {
            "role": "system",
            "content": [
                {
                    "text": 'You are analyzing ChatGPT responses to multiple choice questions. Your task is to extract ChatGPT\'s final answer. \n\nIf you can identify ChatGPT\'s final answer, reply with just that letter inside finalAnswer tags. For example, "<finalAnswer>C</finalAnswer>".\n\n If you cannot identify the answer, reply with "<finalAnswer>Inconclusive</finalAnswer>" ',
                    "type": "text",
                }
            ],
        },
    ]
    if model == Model.GPT3_5:
        extractor_prompt += EXEMPLARS_FOR_GPT3_EXTRACTOR
    elif model == Model.GPT4 or model == Model.GPT4O:
        extractor_prompt += EXEMPLARS_FOR_GPT4_EXTRACTOR
    else:
        print(f"Extracting with chatgpt for model {model} is not supported")
    extractor_prompt += [
        {
            "role": "user",
            "content": f"""<question>{exam_question.format_question()}</question> 
<response>{original_response}</response>""",
        }
    ]

    try:
        response = client.chat.completions.create(
            # for some reason, gpt-4-turbo seems to be better at extraction than gpt-4o.
            # therefore, we will use gpt-4-turbo for extraction.
            model="gpt-4-turbo",
            # model="gpt-3.5-turbo",
            messages=extractor_prompt,
            max_tokens=256,
        )
    except RateLimitError as e:
        print(f"[ERROR] Got RateLimitError with extraction: {e}")
        return original_response, "EXTRACTION_ERROR_RATELIMIT"
    except Exception as e:
        print(f"[ERROR] Got error with extraction: {e}")
        # print("[INFO] prompt used: {}".format(prompt))
        return original_response, "EXTRACTION_ERROR"

    response = response.choices[0].message.content

    pattern = r"<finalAnswer>(.*?)<\/finalAnswer>"
    match = re.search(pattern, response, re.DOTALL)
    extracted_answer = "PARSE_ERROR"
    if match:
        extracted_answer = match.group(1)

    print(f"   used chatgpt to extract answer {extracted_answer}")

    return original_response, extracted_answer


def use_chatgpt_to_extract_answer_textinput_assistants(
    client, exam_question: ExamQuestion, original_response
):
    extractor_prompt = [
        {
            "role": "system",
            "content": 'You are analyzing ChatGPT responses to multiple choice questions. Your task is to extract ChatGPT\'s final answer. \n\nIf you can identify ChatGPT\'s final answer, reply with just that letter inside finalAnswer tags. For example, "<finalAnswer>C</finalAnswer>".\n\nIf you cannot identify the answer, reply with "<finalAnswer>Inconclusive</finalAnswer>" ',
        },
    ]
    extractor_prompt += ASSISTANTS_EXEMPLARS_FOR_EXTRACTOR
    extractor_prompt += [
        {
            "role": "user",
            "content": f"""<question>{exam_question.format_question()}</question> 
<response>{original_response}</response>""",
        }
    ]

    # print(extractor_prompt)
    try:
        response = client.chat.completions.create(
            # Update June 1: there was a significant quality drop in gpt-4-turbo
            # (it couldn't do basic output format of <finalAnswer>) so I'm
            # switching to gpt-4o.
            model="gpt-4o",
            # model="gpt-4-turbo",
            # model="gpt-3.5-turbo",
            messages=extractor_prompt,
            max_tokens=256,
        )
    except RateLimitError as e:
        print(f"[ERROR] Got RateLimitError with extraction: {e}")
        return original_response, "EXTRACTION_ERROR_RATELIMIT"
    except Exception as e:
        print(f"[ERROR] Got error with extraction: {e}")
        # print("[INFO] prompt used: {}".format(prompt))
        return original_response, "EXTRACTION_ERROR"
    response = response.choices[0].message.content

    # print(response)
    pattern = r"<finalAnswer>(.*?)<\/finalAnswer>"
    match = re.search(pattern, response, re.DOTALL)
    extracted_answer = "PARSE_ERROR"
    if match:
        extracted_answer = match.group(1)

    print(f"   used chatgpt to extract answer {extracted_answer}")

    return original_response, extracted_answer
