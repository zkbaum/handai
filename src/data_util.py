"""
Helper functions for reading data from CSVs and parsing into dataclasses.
"""

import pandas as pd
from enum import Enum, auto
import re
from dataclasses import dataclass
from typing import Optional
import math
from private import ROOT_DIR, dic_to_exam_question, dic_to_media


@dataclass
class File:
    """File for reference in openAI storage"""

    file_id: str
    url: str
    citation: str


@dataclass
class Reference:
    """File for reference in openAI storage"""

    question_num: str
    reference_num: str
    openai_file_id: str
    openai_file_name: str
    reference: str
    url: str
    is_uploaded: str
    year: str

    def get_text(self):
        # print(self)
        year = str(self.year).split(".")[0]
        question_num = str(self.question_num).split(".")[0]
        reference_num = str(self.reference_num).split(".")[0]
        root_dir = f"{ROOT_DIR}/data/references/handai-{year}-references/drive"
        path = f"{root_dir}/question_{question_num}/reference_{reference_num}/question_{question_num}_reference_{reference_num}_processed.txt"
        content = ""
        with open(path, "r", encoding="utf-8") as file:
            content = file.read()
        return content


def read_references_as_dict(year: int) -> "dict[int, Reference]":
    path = f"{ROOT_DIR}/data/references/handai-2013-references/2013-references.csv"
    if year == 2012:
        path = f"{ROOT_DIR}/data/references/handai-2012-references/2012-references.csv"

    print(f"reading references at {path}")
    references_list = read_references_csv(path)

    references = {}
    for ref in references_list:
        if ref.question_num not in references:
            references[ref.question_num] = []
        references[ref.question_num].append(ref)

    return references


def read_references_csv(filepath) -> "list[Reference]":
    df = pd.read_csv(filepath)
    dict_list = df.to_dict(orient="records")
    references = []

    year = "2013"
    if "2012" in filepath:
        year = "2012"

    print(f"using year {year}")

    for dic in dict_list:
        question_num = dic["question_num"]
        reference_num = dic["reference_num"]
        openai_file_id = dic["openai_file_id"]
        openai_file_name = dic["openai_file_name"]
        reference = dic["reference"]
        url = dic["Upload link - drive folder"]
        is_uploaded = (
            "Yes" == dic["Did you download the PDF and upload to the drive folder?"]
        )
        references.append(
            Reference(
                question_num=question_num,
                reference_num=reference_num,
                openai_file_id=openai_file_id,
                openai_file_name=openai_file_name,
                reference=reference,
                url=url,
                is_uploaded=is_uploaded,
                year=year,
            )
        )

    return references


class MediaType(Enum):
    IMAGE = "Image"
    VIDEO = "Video"


MEDIA_TYPE_MAP = {member.value: member for member in MediaType}


class Category(Enum):
    ANCILLARY = "Ancillary"
    BASIC_SCIENCE = "Basic Science"
    BONE_AND_JOINT = "Bone and Joint"
    MISC = "Misc"
    NEUROMUSCULAR = "Neuromuscular"
    SKIN = "Skin"
    UNSPECIFIED = "Unspecified"
    VASCULAR = "Vascular"


CATEGORY_MAP = {member.value: member for member in Category}


# really crazy hack to get around the the nbsp issue
def _remove_nbsp_first_30_chars(input_string):
    # Split the string into the first 30 characters and the rest
    first_30 = input_string[:30]
    rest = input_string[30:]

    # Replace '&nbsp;' with '' in the first 30 characters
    first_30_cleaned = first_30.replace("&nbsp;", "")

    # Concatenate the cleaned first 30 characters with the rest of the string
    cleaned_string = first_30_cleaned + rest

    return cleaned_string


# This is a hack...need to figure out a better way of handling this.
def _is_nan(txt):
    try:
        float_value = float(txt)
        return math.isnan(float_value)
    except ValueError:
        # Handle the case where the conversion to float fails
        # (if 'value' is not a valid number)
        return False


@dataclass
class ExamQuestion:
    """Class for a question on the Exam."""

    question_id: str  # maybe should be int? keep everything consistent
    title: str
    category: Category
    objective: str
    question: str  # was: stem
    lead_in: str
    commentary: str
    reference: str
    question_status: str
    author: str
    creation_date: str
    last_modified: str
    keywords: str
    origination_exam: str
    question_rating: float
    note: str
    question_year: str
    question_type: str
    is_link_question: bool
    fk_question_series_id: str
    sequence_in_question_series: str
    remediation_field_1: str
    remediation_field_2: str
    fixed_answer_option_sequence: bool
    correct_answer: str
    choice_a: str
    choice_b: str
    choice_c: str
    choice_d: str
    choice_e: str
    media: "list[ExamMedia]"
    correct_answer_percentage: float
    distractor_percentages: "dic[str, float]"
    references: "list[Reference]"

    def attach_reference(self, ref: Reference):
        """
        This is REALLY hacky. but since the references are created
        after exam questions...we will manually add them after
        """
        if ref.is_uploaded:
            # print(f'attaching reference {ref} to this question {self.get_question_number()}')
            self.references.append(ref)

    def get_references(self):
        return self.references

    def get_clean_commentary(self) -> str:
        pattern = r"Prefer+ed Response: ?[A-Z](<br /><br />|\s*)(.*)"

        # Use re.search() to find the text after the matched pattern
        match = re.search(
            pattern,
            _remove_nbsp_first_30_chars(self.commentary),
            flags=re.IGNORECASE | re.DOTALL,
        )
        if match:
            # Return everything after the "Preferred Response"
            return match.group(2)  # group(2) refers to the (.*) part of the pattern
        else:
            print(
                "[WARNING] Preferred response not found in commentary: ",
                self.commentary,
            )
            # If there's no "Preferred Response", return the original commentary
            return self.commentary

    def format_question(self) -> str:
        question = f"{self.question}\n"
        # TODO(zkbaum) need a better way of handling missing questions...
        if not _is_nan(self.choice_a):
            question += f"A. {self.choice_a}\n"
        if not _is_nan(self.choice_b):
            question += f"B. {self.choice_b}\n"
        if not _is_nan(self.choice_c):
            question += f"C. {self.choice_c}\n"
        if not _is_nan(self.choice_d):
            question += f"D. {self.choice_d}\n"
        if not _is_nan(self.choice_e):
            question += f"E. {self.choice_e}"
        return question

    def get_year(self) -> int:
        # text = "2013 Self-Assessment Examination"
        match = re.search(r"\b\d{4}\b", self.origination_exam)

        if match:
            year = match.group()
            return int(year)
        else:
            print("No year found in the text: ", self.origination_exam)
            return None

    def get_question_number(self) -> int | None:
        # Use regular expression to find the number after 'Q'
        match = re.search(r"Q(\d+)", self.title, flags=re.IGNORECASE)

        if match:
            return int(match.group(1))
        else:
            print("No question number found in :", self.title)
            return None

    def get_correct_answer(self):
        return ["A", "B", "C", "D", "E"][int(self.correct_answer)]

    def question_has_text_only(self):
        return len([m for m in self.media if m.show_in_question]) == 0

    def question_has_text_and_images(self):
        return len([m for m in self.media if m.show_in_question]) > 0

    def commentary_has_text_only(self):
        return len([m for m in self.media if m.show_in_commentary]) == 0

    def commentary_has_text_and_images(self):
        return len([m for m in self.media if m.show_in_commentary]) > 0

    def get_question_content_type(self):
        if self.question_has_text_only():
            return ContentType.TEXT_ONLY
        return ContentType.TEXT_AND_IMAGES

    def get_human_distribution(self):
        distribution = self.distractor_percentages
        letter_to_text = {
            "A": self.choice_a,
            "B": self.choice_b,
            "C": self.choice_c,
            "D": self.choice_d,
            "E": self.choice_e,
        }
        # hack to account for sometimes E not being there
        if _is_nan(self.choice_e):
            distribution[self.choice_e] = 0

        answer_text = letter_to_text[self.get_correct_answer()]
        distribution[answer_text] = f"{self.correct_answer_percentage}%"
        return {
            "A": distribution[self.choice_a],
            "B": distribution[self.choice_b],
            "C": distribution[self.choice_c],
            "D": distribution[self.choice_d],
            "E": distribution[self.choice_e],
        }


@dataclass
class ExamMedia:
    """Class for the media related to an exam question."""

    question_id: str
    asset_id: str
    asset_title: str
    description: str
    media_type: MediaType
    media_file_name: str
    keyword: str
    note: str
    caption: str
    figure_title: str
    show_in_question: bool
    show_in_commentary: bool
    relative_file_path: str
    file_name: str

    def get_url(self):
        ext = self.file_name.split(".").pop()
        if ext == "bmp":
            hash = self.file_name.split(".")[0]
            return f"https://handai.s3.us-east-2.amazonaws.com/{hash}.png"
        relative_path = self.relative_file_path.replace("\\", "/")
        return f"https://de90rgl81jte0.cloudfront.net{relative_path}"


def read_questions_csv(
    filepath: str, question_id_to_media: "dict[str, list[ExamMedia]]"
) -> "list[ExamQuestion]":
    df = pd.read_csv(filepath)
    dict_list = df.to_dict(orient="records")

    questions = []
    for dic in dict_list:
        question_id = dic["QuestionID"]

        # extract media if it it exists for this question
        media = []
        if question_id in question_id_to_media:
            media = question_id_to_media[question_id]

        question = dic_to_exam_question(dic, CATEGORY_MAP, media)
        questions.append(question)

    # VALIDATION to catch any obvious formatting problems
    for q in questions:
        q.get_year()
        q.get_question_number()
        # q.get_clean_commentary()

    return questions


def read_media_csv(filepath) -> "list[ExamMedia]":
    df = pd.read_csv(filepath)
    dict_list = df.to_dict(orient="records")

    question_to_media = {}
    for dic in dict_list:
        question_id = dic["QuestionID"]
        if question_id not in question_to_media:
            question_to_media[question_id] = []
        media = dic_to_media(dic, question_id, MEDIA_TYPE_MAP)
        question_to_media[question_id].append(media)

    return question_to_media


class ContentType(Enum):
    TEXT_ONLY = auto()
    TEXT_AND_IMAGES = auto()


class QuestionsBuilder:
    question_id: str
    year: Optional[int]
    question_content_type: Optional[ContentType]
    commentary_content_type: Optional[ContentType]

    def year(self, year: int):
        self.year = year
        return self

    def question_content_type(self, question_content_type: ContentType):
        self.question_content_type = question_content_type
        return self

    def commentary_content_type(self, commentary_content_type: ContentType):
        self.commentary_content_type = commentary_content_type
        return self

    def build(self) -> "list[ExamQuestion]":
        media = read_media_csv(f"{ROOT_DIR}/data/assh-data/media.csv")
        questions = read_questions_csv(
            f"{ROOT_DIR}/data/assh-data/questions.csv", media
        )

        # filter out any questions with videos
        questions = [
            q
            for q in questions
            if len([m for m in q.media if m.media_type == MediaType.VIDEO]) == 0
        ]

        # filter out based on year
        if self.year:
            questions = [q for q in questions if q.get_year() == self.year]

        # filter based on content restrictions - question
        if self.question_content_type == ContentType.TEXT_ONLY:
            questions = [q for q in questions if q.question_has_text_only()]
        elif self.question_content_type == ContentType.TEXT_AND_IMAGES:
            questions = [q for q in questions if q.question_has_text_and_images()]

        # filter based on content restrictions - commentary
        if self.commentary_content_type == ContentType.TEXT_ONLY:
            questions = [q for q in questions if q.commentary_has_text_only()]
        elif self.commentary_content_type == ContentType.TEXT_AND_IMAGES:
            questions = [q for q in questions if q.commentary_has_text_and_images()]

        # sort by year and question number
        questions.sort(
            key=lambda question: (
                question.get_year(),
                question.get_question_number(),
            )
        )

        return questions


def get_n_examples_from_each_category(exam_questions, n, categories: "list[Category]"):
    category_ct = {}
    ret = []
    filtered_questions = [q for q in exam_questions if q.category in categories]

    # Distribute the questions evenly across each category.
    for entry in filtered_questions:
        if entry.category not in category_ct:
            category_ct[entry.category] = 1
            ret.append(entry)
            continue

        if category_ct[entry.category] < n:
            category_ct[entry.category] += 1
            ret.append(entry)

    # print("[DEBUG] examplar distribution: {}".format(category_ct))
    return ret


def get_knn_exemplars(exam_questions: "list[ExamQuestion]", n: int, category: Category):
    filtered_questions = [q for q in exam_questions if q.category == category]

    # If we have >= n train examples in this category, just use  the first n.
    if len(filtered_questions) >= n:
        return filtered_questions[0:n]

    # If we have <n, pull from other cateogries until we reach n
    catagories = [c for c in Category]
    catagories.remove(category)
    buffer = get_n_examples_from_each_category(exam_questions, 1, catagories)

    return filtered_questions + buffer[0 : n - len(filtered_questions)]


def attach_references(references_dic, entry):
    question_num = entry.get_question_number()
    if question_num in references_dic:
        documents = references_dic[question_num]
        for doc in documents:
            entry.attach_reference(doc)
        print(
            f"   attached {len(entry.references)} of {len(documents)} "
            f"references as documents for question {question_num}"
        )


def prune_questions_without_any_references(
    exam_questions: "list[ExamQuestion]", year: int
):
    """
    For some retreival experiments, we only want to consider questions that
    have at least 1 reference available.
    """
    references_dic = read_references_as_dict(year)

    # Only consider questions with references
    print(f"before pruning questions without references: n={len(exam_questions)}")
    before_set = [x.get_question_number() for x in exam_questions]
    for entry in exam_questions:
        attach_references(references_dic, entry)
    pruned = [
        x
        for x in exam_questions
        if len(x.references) > 0 and not x.question_has_text_and_images()
    ]
    print(f"after pruning: n={len(pruned)}")
    after_set = [x.get_question_number() for x in pruned]
    print(f"removed {sorted(set(before_set)-set(after_set))}")

    return pruned
