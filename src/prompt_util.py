
"""
Helper functions to construct prompts.
"""
from data_util import Reference, Category, ExamQuestion, MediaType, QuestionsBuilder, get_n_examples_from_each_category, ContentType
from openai import OpenAI   

def _generate_fake_exemplar(question, choice_a, choice_b, choice_c, choice_d, choice_e, correct_answer, commentary):
    answer_to_num = {
        'A':0,
        'B':1,
        'C':2,
        'D':3,
        'E':4
    }
    return ExamQuestion(
        question=question, 
        choice_a=choice_a, 
        choice_b=choice_b, 
        choice_c=choice_c, 
        choice_d=choice_d, 
        choice_e=choice_e, 
        correct_answer=answer_to_num[correct_answer],
        question_id = None,
        title = None,
        category = None,
        objective = None,
        lead_in = None,
        commentary = commentary,
        reference = None,
        question_status = None,
        author = None,
        creation_date = None,
        last_modified = None,
        keywords = None,
        origination_exam = None,
        question_rating = None,
        note = None,
        question_year = None,
        question_type = None,
        is_link_question = None,
        fk_question_series_id = None,
        sequence_in_question_series = None,
        remediation_field_1 = None,
        remediation_field_2 = None,
        fixed_answer_option_sequence = None,
        media=[],
        correct_answer_percentage = None,
        distractor_percentages = None,
        references = []
    )

def get_no_prompt_exemplars():
    """
    To simulate the "no prompt" case, we need to give a simple example enforce some formatting. 
    """    
    return [
         _generate_fake_exemplar(
            question="What is 1+1?",
            choice_a='0',
            choice_b='1',
            choice_c='2',
            choice_d='3',
            choice_e='4',
            correct_answer='C',
            commentary="Preferred Response: C<br /><br />This is some explanation about why 1+1=2"
        ),
    ]
    
def _create_question_content(
        exam_question: ExamQuestion, 
        include_reference_text: bool = False, 
        include_question_tag: bool = False
):
    content = []
       
    if include_reference_text:
        ref_texts = []
        n = 0
        for ref in exam_question.references:
            text = ref.get_text()
            # Hack: for train exemplars, only take first 1000 chars, otherwise we will hit token limit
            if exam_question.get_year() == 2012:
                text = text[0:1000]+' ... (rest of document removed because this is an examplar)'
            ref_texts.append(f"<document{n}>{text}</document{n}>")
            n += 1
        if n == 0:
            ref_texts.append("NONE")


        content.append(
            {
                "type": "text", 
                "text": f"Base your response on the following document(s): {"".join(ref_texts)}"
            }
        )

    prompt = exam_question.format_question()
    if include_question_tag:
        prompt = f'<question>{prompt}</question>'
    content.append(
        {
            "type": "text", 
            "text": prompt
        }
    )
    
    # For questions with images, also attach the images.
    for media in exam_question.media:
        if media.show_in_question and media.media_type == MediaType.IMAGE:
            if include_question_tag:
                content.append({
                    "type": "text", 
                    "text": f'Figure {media.figure_title}',
                })
            content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": media.get_url()
                    },
                },
            )

    if include_reference_text:
        content.append(
            {
                "type": "text", 
                "text": "REMINDER: make sure to include <discussion></discussion> and <answer></answer> tags"
            }
        )
        
    return content


def _create_discussion_content(exam_question: ExamQuestion, include_reference_text: bool = False):
    ret = ""
    discussion = exam_question.get_clean_commentary()
    if include_reference_text:
        if exam_question.references:
            discussion = "Based on the provided documents, "+discussion
        else:
            discussion = "I did not receive any documents, so I will base my answer on my expert medical knowledge. "+discussion
    ret += f'<discussion>{discussion}</discussion>'
    ret += f'<answer>{exam_question.get_correct_answer()}</answer>'
    return ret

def create_prompt(
        preamble: str, 
        exemplars: "list[ExamQuestion]", 
        exam_question: ExamQuestion
):
    """
    Creates a prompt based on a preamble, list of exemplars, and question to ask.
    This supports both text only and image prompts.
    
    Returns
        inputs - message list of everything up until answer (preamble, examplars, question)
        target - message list of expected completion - e.g. discussion  and answer
    """
    inputs = []

    if preamble is not None:
        # Add preamble
        inputs = [
            {
                "role": "system",
                "content": preamble
            }
        ]

    is_few_shot = False
    # Add examplars
    if exemplars:
        for exemplar in exemplars:
            is_few_shot = True
            inputs.append(
                {
                    "role": "user",
                    "content": _create_question_content(
                        exemplar,
                        include_question_tag=is_few_shot
                    )
                }
            )
            inputs.append(
                {
                    "role": "assistant",
                    "content": _create_discussion_content(exemplar)
                }
            )

    # Add question
    inputs.append(
        {
            "role": "user",
            "content": _create_question_content(
                exam_question,
                include_question_tag=is_few_shot
            )
        }
    )
    
    target = [
        {
            "role": "assistant",
            "content": _create_discussion_content(exam_question)
        }
    ]

    return inputs, target



def create_instructions_for_assistant():
    train_set = (QuestionsBuilder()
                  .year(2008)
                  .question_content_type(ContentType.TEXT_ONLY)
                  .commentary_content_type(ContentType.TEXT_ONLY)
                  .build())

    exemplar_text = ''
    exemplars = get_n_examples_from_each_category(train_set, 1, list(Category))
    for ex in exemplars:
        as_txt = f"""<question>{ex.format_question()}</question>
        
<discussion>{ex.get_clean_commentary()}</discussion>

<answer>{ex.get_correct_answer()}</answer>

"""
        exemplar_text += as_txt


    instructions = f"""You are a board certified hand surgeon. \
You are taking a multiple choice exam to test your hand surgery knowledge. \
You will be presented with a question and a few possible answer choices. 

First, search through your knowledge base to find information about the \
question. Then, inside <discussion></discussion> tags, briefly discuss \
each option. Finally, decide on the correct answer (citing sources from \
the knowledge base) and write the final answer inside <answer></answer> \
tags. Even if you are unsure, please pick one letter you are most \
confident about.

Please make sure to search every file in your knowledge base. \
If you cannot find the information in the knowledge base, fall back to your \
general medical knowledge. Here are examples of the desired output.

{exemplar_text}
"""
    return instructions
