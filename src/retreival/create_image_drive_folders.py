"""Since I manually unfolded the references for images, here's a different script to create the dirve folders."""

import csv
from create_drive_folders import write_directories, DriveDirectory


# Define the path to the CSV file (the sheet of unfolded questions)
csv_file_path = "PATH_TO_UNFOLDED_QUESTIONS"

# Initialize a list to hold the parsed data
parsed_data = []

# Read the CSV file and parse the data
with open(csv_file_path, mode="r", encoding="utf-8") as file:
    csv_reader = csv.DictReader(file)
    for row in csv_reader:
        question_num = row["question_num"]
        reference_num = row["reference_num"]
        reference = row["reference"]
        parsed_data.append((int(question_num), int(reference_num), reference))

structure = {}
for question_num, reference_num, reference in parsed_data:
    if question_num not in structure:
        structure[question_num] = []
    structure[question_num].append(reference_num)

created_directories = write_directories(structure, "2013 References - Image")

for dir in created_directories:
    print(
        f"{dir.question_num}###{dir.reference_num}###https://drive.google.com/drive/folders/{dir.folder_id}"
    )
