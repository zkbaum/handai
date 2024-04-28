"""
Since each v1 assistant can only support 20 files, we are creating an assistant
for each question. However, this is no longer required in assistants v2 because
of the increased file limit. So this script can be mostly ignored.
"""

from data_util import (
    Category,
    Reference,
    read_references_csv,
    ExamQuestion,
    QuestionsBuilder,
    ContentType,
    get_n_examples_from_each_category,
    get_knn_exemplars,
)
from prompt_util import create_instructions_for_assistant
from inference_util import (
    Model,
    do_chat_completion,
    parse_response_string_answeronly,
    parse_response_answeronly,
    write_inference_csv,
    InferenceResult,
)
from openai import OpenAI
import time
import random
from private import ROOT_DIR

_MAX_FILES_PER_ASSISTANT = 20

client = OpenAI()
references_list = read_references_csv(
    f'{ROOT_DIR}/data/references/handai-2013-references/2013-references.csv"
)

# Normalize references
references = {}
all_file_ids = []
for ref in references_list:
    if ref.question_num not in references:
        references[ref.question_num] = []
    references[ref.question_num].append(ref)
    if ref.is_uploaded:
        all_file_ids.append(ref.openai_file_id)


def _create_assistant_deprecated():
    """DEPRECATED.

    Originally, I thought we'd partition everything into groups of 20 questions.
    However, now I'm thinking it'd be simpler to just give 1 assistant per question.
    If needed, we can add random docs for noise.
    """

    questions_selected = []

    num_files = 0
    for question_num in references:
        num_references = len(references[question_num])
        num_uploaded = len(
            [0 for ref in references[question_num] if ref.is_uploaded]
        )
        print(
            f"question_{question_num}: {num_references} total, {num_uploaded} uploaded"
        )
        if num_uploaded == num_references:
            # If adding this will be within the limit, add it
            if num_files + num_references <= MAX_FILES_PER_ASSISTANT:
                num_files += num_references
                questions_selected.append(question_num)
                print(
                    f"We will use this for the assistant! Current file count: {num_files}"
                )
            # otherwise, this must go into the new thing
            else:
                print(
                    f"Cannot fit {num_references} more files (current count {num_files})"
                )
                print(
                    "This is the end of this assistant, eventually you should use this to start the next assistant"
                )
                break
        else:
            print("skipping")

    print(f"creating assistant with questions: {questions_selected}")
    print("this will take a minute...")

    file_ids = []
    for q in questions_selected:
        file_ids += [ref.openai_file_id for ref in references[q]]

    questions_selected = [str(x) for x in questions_selected]
    first_question = questions_selected[0]
    last_question = questions_selected[len(questions_selected) - 1]

    assistant = client.beta.assistants.create(
        name=f"handai_q{first_question}_thru_q{last_question}",
        description="a partition of the handai assistant for a research project",
        instructions=create_instructions_for_assistant(),
        model="gpt-4-turbo-preview",
        tools=[{"type": "retrieval"}],
        file_ids=file_ids,
        metadata={"questions_selected": ",".join(questions_selected)},
    )

    print(f"created assistant: {assistant.id}")


def _create_assisstant(
    question_num: str, references: "list[Reference]", all_file_ids: "list[str]"
):
    """
    Creates an assistant for the given question
    """

    # Pad up to file limit
    target_file_ids = [ref.openai_file_id for ref in references]
    other_ids = [id for id in all_file_ids if id not in target_file_ids]
    padding_len = _MAX_FILES_PER_ASSISTANT - len(target_file_ids)
    target_file_ids_with_padding = target_file_ids + random.sample(
        other_ids, padding_len
    )

    assistant = client.beta.assistants.create(
        name=f"handai_q{question_num}",
        description=f"a handai assistant for question {question_num}",
        instructions=create_instructions_for_assistant(),
        model="gpt-4-turbo-preview",
        tools=[{"type": "retrieval"}],
        file_ids=target_file_ids_with_padding,
        metadata={"target_question": question_num, "version": "alpha"},
    )
    return assistant


def _delete_assistants(client, questions_to_delete, dry_run=True):
    assistants = client.beta.assistants.list()
    for a in assistants:
        if "target_question" in a.metadata:
            question_num = int(a.metadata["target_question"])
            if question_num in questions_to_delete:
                print(f"deleting asssistant {a.id} with metadata {a.metadata}")
                if not dry_run:
                    resp = client.beta.assistants.delete(a.id)
                    print(f"deleted with response {resp}")


assistants_to_update = [154, 165, 178]
# _delete_assistants(client, assistants_to_update, dry_run=False)

# Create 1 assistant per question.
a_count = 0
for question_num, ref_list in references.items():
    if question_num not in assistants_to_update:
        continue
    available_refs = [ref for ref in ref_list if ref.is_uploaded]
    num_references = len(ref_list)
    num_available_refs = len(available_refs)
    print(
        f"question {question_num}: {num_references} total, {num_available_refs} available"
    )
    if num_available_refs / num_references >= 0.5:
        print(f"  creating assistant for question {question_num}")
        assistant = _create_assisstant(
            question_num, available_refs, all_file_ids
        )
        print(f"  created asssistant {assistant.id}")
        a_count += 1
    else:
        print(
            f"  not enough references available. skipping question {question_num}"
        )


# If we only create assistant when 100 is available, we get 84/141
# If we do it when >= 50% of sources are there, we get 132/141
# If >50%, then 105/141
print(f"created {a_count} asssistants.")

print("done :)")
