"""Ingests the references as files to open AI api
"""

import os
import sys
import pandas as pd
from dataclasses import dataclass
from openai import OpenAI
from datetime import datetime
import csv

# Hack to import from parent dir
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
from private import ROOT_DIR


@dataclass
class Reference:
    """File for reference before uploading to openAI"""

    question_id: str
    question_num: str
    reference_num: str
    reference: str
    is_uploaded_to_drive: bool


@dataclass
class OpenAIFile:
    """Wrapper for once the file is stored in openai storage"""

    reference: Reference
    file_id: str
    file_name: str


def read_input_references_csv(filepath) -> "dic[str, dic[str, Reference]]":
    df = pd.read_csv(filepath)
    dict_list = df.to_dict(orient="records")
    # { 1: {1: Reference(), 2: Reference()}, 34: {1: Reference()}}
    references = {}
    for dic in dict_list:
        # question_id = dic["question_id"]
        question_num = dic["question_num"]
        reference_num = dic["reference_num"]
        reference = dic["reference"]
        is_uploaded = (
            "Yes" == dic["Did you download the PDF and upload to the drive folder?"]
        )
        if question_num not in references:
            references[question_num] = {}
        if reference_num in references[question_num]:
            print("FATAL ERROR: UNEXPECTED DUP REFERENCE")
            exit()
        references[question_num][reference_num] = Reference(
            question_id="n/a",
            question_num=question_num,
            reference_num=reference_num,
            reference=reference,
            is_uploaded_to_drive=is_uploaded,
        )

    return references


def get_single_file_name(directory):
    files = os.listdir(directory)
    files = [f for f in files if not f.startswith(".")]
    if len(files) == 1:
        return files[0]
    elif len(files) > 1:
        print(f"WARNING: got more than 1 file: {files}")
        return None
    else:
        return None


def _upload_file_to_openai(client, root_dir_path: str, ref: Reference):
    # Find the directory
    dir_path = (
        f"{root_dir_path}/question_{ref.question_num}/reference_{ref.reference_num}"
    )
    file_path = f"{dir_path}/{get_single_file_name(dir_path)}"
    if file_path is None:
        print(f"FATAL ERROR, {dir_path} should have exactly 1 file")
        exit(1)

    # rename the file to something easier to read
    new_file_path = (
        f"{dir_path}/question_{ref.question_num}_reference_{ref.reference_num}.pdf"
    )
    os.rename(file_path, new_file_path)

    # We expect there to be one fine in a directory
    openai_file = client.files.create(
        file=open(new_file_path, "rb"),
        purpose="assistants",
    )
    print(f"uploaded {file_path} as {openai_file.id}")
    return openai_file


def _upload_files_to_openai(client, root_dir_path: str, input) -> "list[OpenAIFile]":
    results = []
    for question_num in input:
        for ref_num in input[question_num]:
            ref = input[question_num][ref_num]
            openai_filename = ""
            openai_fileid = ""
            # TODO(zkbaum) add caching so that if already exists, don't upload
            if ref.is_uploaded_to_drive:
                openai_file = _upload_file_to_openai(client, root_dir_path, ref)
                openai_filename = openai_file.filename
                openai_fileid = openai_file.id
            results.append(
                OpenAIFile(
                    reference=ref,
                    file_id=openai_fileid,
                    file_name=openai_filename,
                )
            )
    # print(results)
    return results


def _write_output_csv(filepath, results):

    with open(filepath, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "question_num",
                "reference_num",
                "openai_file_id",
                "openai_file_name",
            ]
        )

        for res in results:
            row = [
                res.reference.question_num,
                res.reference.reference_num,
                res.file_id,
                res.file_name,
            ]
            writer.writerow(row)
    print(f"wrote results to: {filepath}")


def _read_manual_references():
    return {
        154: {
            1: Reference(
                question_id="1114",
                question_num="154",
                reference_num="1",
                reference="Lineaweaver WC, Hill MK, Buncke GM, Follansbee S, Buncke HJ, Wong RK, Manders EK, Grotting JC, Anthony J, Mathes SJ. Aeromonas hydrophila infections following use of medicinal leeches in replantation and flap surgery. Ann Plast Surg. 1992;29(3):238-44.",
                is_uploaded_to_drive=True,
            )
        },
        165: {
            4: Reference(
                question_id="1114",
                question_num="165",
                reference_num="4",
                reference="Manktelow RT, Zuker RM. The principles of functioning muscle transplantation: Applications to the upper arm. Ann Plast Surg. 1989;22:275-282.",
                is_uploaded_to_drive=True,
            )
        },
        178: {
            2: Reference(
                question_id="1138",
                question_num="178",
                reference_num="2",
                reference="Mackay DR, Manders EK, Saggers GC, Banducci DR, Prinsloo J, Klugman K. Aeromonas species isolated from medicinal leeches. Ann Plast Surg. 1999;42(3):275-279.",
                is_uploaded_to_drive=True,
            )
        },
    }


def _delete_all_files(client, dry_run: bool):
    try:
        # List all files
        files = client.files.list()

        # Loop through the files and delete them
        for file in files.data:
            file_id = file.id
            filename = file.filename
            delete_cutoff = datetime(2024, 6, 1, 12)
            if file.created_at < delete_cutoff.timestamp():
                print(f"skipping {filename} because before {delete_cutoff}")
            elif dry_run:
                print(
                    f"[dry run] would have Deleted file {filename} with ID: {file_id}"
                )
            else:
                client.files.delete(file_id)
                print(f"Deleted file {filename} with ID: {file_id}")
    except Exception as e:
        print(f"An error occurred: {e}")
        print("retrying...")
        _delete_all_files(client, dry_run=dry_run)


def _validate_pdfs(root_dir_path: str) -> bool:
    """
    Validates that all subdirectories in the given directory path have at most 1 file.

    Args:
    - root_dir_path (str): The path to the directory to validate.

    Returns:
    - bool: whether the directory structure is valid
    """

    # Walk through the directory structure
    for root, dirs, files in os.walk(root_dir_path):
        # Skip the root directory
        if root == root_dir_path:
            continue

        # Check the number of files in each subdirectory
        files = [f for f in files if not f.startswith(".")]
        if len(files) > 1:
            print(f"[ERROR] Found invalid directory: {root} with files {files}")
            exit()

    print("local files are valid :)")


# input_references_path = f"{ROOT_DIR}/data/references/handai-2013-references/INPUT.csv"
# input = read_input_references_csv(input_references_path)
input = _read_manual_references()

PDF_DIRECTORY_PATH = f"{ROOT_DIR}/data/references/handai-2013-references/drive"
_validate_pdfs(PDF_DIRECTORY_PATH)

client = OpenAI()
results = _upload_files_to_openai(client, PDF_DIRECTORY_PATH, input)
exit()

current_timestamp = datetime.now()
formatted_timestamp = current_timestamp.strftime("%Y-%m-%d %H:%M:%S")
filepath = f"{ROOT_DIR}/out/files/files-{formatted_timestamp}.csv"
_write_output_csv(filepath, results)
