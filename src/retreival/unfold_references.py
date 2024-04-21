"""
Helper script to create a sheet with 1 row per question/reference.
Then also creates a google drive folder for each row. That way it's easier
to keep track of references during manual entry.
"""

import re
from datetime import datetime
import csv
from create_drive_folders import write_directories, DriveDirectory
from data_util import (
    Category,
    ExamQuestion,
    QuestionsBuilder,
    ContentType,
    get_n_examples_from_each_category,
    get_knn_exemplars,
)
from dataclasses import dataclass
from private import ROOT_DIR


@dataclass
class UnfoldedReference:
    """Parsed reference"""

    original_question: ExamQuestion
    reference_num: str
    unfolded_reference: str


def _unfold_references(references: str):
    references_with_empty_divs_corrected = references.replace(
        "<!-- reference -->", ""
    ).split("<br />")

    # Remove <div> and </div> tags and filter out empty references
    references_with_empty_divs_corrected = [
        ref.replace("<div>", "").replace("</div>", "").strip()
        for ref in references_with_empty_divs_corrected
        if ref.strip()
    ]

    # Replace double single quotes with a single quote and filter out any empty strings
    references_with_empty_divs_corrected = [
        ref for ref in references_with_empty_divs_corrected if ref
    ]

    references_filtered = [
        re.sub("<[^>]+>", "", ref).strip()
        for ref in references_with_empty_divs_corrected
        if not re.sub("<[^>]+>", "", ref).strip() == ""
    ]

    return references_filtered


def _unfold_all_references(eval_set) -> "List[UnfoldedReference]":
    results = []
    for q in eval_set:
        for ref in _unfold_references(q.reference):
            # in the form 1. reference....so split the 1 and reference
            split_text = ref.split(
                ". ", 1
            )  # Limit split to only the first occurrence
            result = (
                [split_text[0], split_text[1]] if len(split_text) == 2 else []
            )
            ref_num = result[0]
            unfolded_ref = result[1]
            try:
                ref_num = int(ref_num)
                # print("x is a valid integer")
            except ValueError:
                print(
                    f"[WARNING] in question {q.get_question_number()}, `{ref_num}` is not a valid integer, skipping"
                )
                continue

            results.append(
                UnfoldedReference(
                    original_question=q,
                    reference_num=ref_num,
                    unfolded_reference=unfolded_ref,
                )
            )
    return results


def _write_unfolded_references(
    results: "list[UnfoldedReference]",
    created_directories: "list[DriveDirectory]",
):
    current_timestamp = datetime.now()
    formatted_timestamp = current_timestamp.strftime("%Y-%m-%d %H:%M:%S")
    filepath = f"{ROOT_DIR}/out/references/references-{formatted_timestamp}.csv"

    # transform created_directories to a form that's easy to search
    drive_directories = {}
    for d in created_directories:
        if d.question_num not in drive_directories:
            drive_directories[d.question_num] = {}
        drive_directories[d.question_num][d.reference_num] = d.folder_id

    # Open the file in write mode ('w') and create a csv.writer object
    with open(filepath, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "question_id",
                "question_num",
                "raw_references",
                "reference_num",
                "unfolded_reference",
                "folder_link",
            ]
        )

        for res in results:
            question_number = res.original_question.get_question_number()
            reference_number = res.reference_num
            row = [
                res.original_question.question_id,
                question_number,
                res.original_question.reference,
                reference_number,
                res.unfolded_reference,
                f"https://drive.google.com/drive/folders/{drive_directories[question_number][reference_number]}",
            ]
            writer.writerow(row)

    print(f"Data has been written to {filepath}")


eval_set = (
    QuestionsBuilder()
    .year(2013)
    .question_content_type(ContentType.TEXT_ONLY)
    .commentary_content_type(ContentType.TEXT_ONLY)
    .build()
)

unfolded = _unfold_all_references(eval_set)

created_directories = []
WRITE_DRIVE = True
if WRITE_DRIVE:
    structure = {}
    for res in unfolded:
        question_num = res.original_question.get_question_number()
        if question_num not in structure:
            structure[question_num] = []
        structure[question_num].append(res.reference_num)

    created_directories = write_directories(structure, "2013 References")


_write_unfolded_references(unfolded, created_directories)
