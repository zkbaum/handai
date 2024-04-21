"""
Merges multiple PDFs into a few small PDFs. This was relevant for assistants v1,
which only supported a 20 files per assistant.
"""

import os
import PyPDF2
from pathlib import Path
from private import ROOT_DIR


def merge_pdfs(paths, output):
    pdf_writer = PyPDF2.PdfWriter()

    for path in paths:
        pdf_reader = PyPDF2.PdfReader(path)
        for page in range(len(pdf_reader.pages)):
            pdf_writer.add_page(pdf_reader.pages[page])

    with open(output, "wb") as out:
        pdf_writer.write(out)


def find_pdfs(start_dir):
    for root, dirs, files in os.walk(start_dir):
        for file in files:
            if file.endswith(".pdf"):
                yield os.path.join(root, file)


def group_directories(base_dir):
    dirs = [
        d
        for d in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, d))
        and d.startswith("question_")
    ]
    dirs.sort(
        key=lambda x: int(x.split("_")[1])
    )  # Sort directories by the number part

    grouped = []
    for i in range(0, len(dirs), 50):
        grouped.append(dirs[i : i + 50])

    return grouped


def main():
    base_dir = f"{ROOT_DIR}/data/references/handai-2013-references/drive/q151_200"  # Change this to your directory path
    grouped_dirs = group_directories(base_dir)

    for index, group in enumerate(grouped_dirs, start=1):
        pdf_paths = []
        for dir_name in group:
            dir_path = os.path.join(base_dir, dir_name)
            pdf_paths.extend(find_pdfs(dir_path))

        if pdf_paths:
            output_path = os.path.join(
                base_dir,
                f'merged_questions_{group[0].split("_")[1]}_to_{group[-1].split("_")[1]}.pdf',
            )
            merge_pdfs(pdf_paths, output_path)
            print(f"Merged PDF saved to: {output_path}")
        else:
            print(f"No PDFs found in directories: {group}")


if __name__ == "__main__":
    main()
