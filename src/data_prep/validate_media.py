"""
Random script to make sure the media has been parsed properly.
"""

from data_util import QuestionsBuilder, ContentType

for year in [2008, 2009, 2010, 2011, 2012, 2013]:
    eval_set = QuestionsBuilder().year(year).build()

    global_media = []
    for q in eval_set:
        for m in q.media:
            if m.asset_id in global_media:
                print(
                    f"[{year}] WARNING: got duplicate media for question {q.get_question_number()} figure {m.figure_title} (id: {m.asset_id})"
                )
            else:
                global_media.append(m.asset_id)
    print(f"{year} has {len(global_media)} assets")
print("media is valid :)")


# validate question distribution
text_count = 0
image_count = 0
category_counts = {}

for year in [2008, 2009, 2010, 2011, 2013]:

    eval_set = QuestionsBuilder().year(year).build()
    for q in eval_set:
        if q.get_question_content_type() == ContentType.TEXT_ONLY:
            text_count += 1
        else:
            image_count += 1

        if q.category not in category_counts:
            category_counts[q.category] = 0
        category_counts[q.category] += 1
    print(year)
print(f"num text questions: {text_count}")
print(f"num image questions: {image_count}")
print(category_counts)
