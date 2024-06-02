"""OpenAI recently released assistants v2. So we will try this
for retrieval as well.

The main differences between v1 and v2 are described here:
https://platform.openai.com/docs/assistants/migration
"""

import os
import sys
import pandas as pd
from openai import OpenAI
import datetime


# Hack to import from parent dir
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
from private import ROOT_DIR
from prompt_util import create_instructions_for_assistant


def _create_vector_store(client):
    # We already uploaded all the files and put the names into a CSV.
    # Therefore, we will simply use those when creating the stores.
    # If we were starting from scratch, we would use batch uploading:
    # https://platform.openai.com/docs/assistants/tools/file-search/step-2-upload-files-and-add-them-to-a-vector-store
    input_references_path = (
        f"{ROOT_DIR}/data/references/handai-2013-references/2013-references.csv"
    )
    df = pd.read_csv(input_references_path)["openai_file_id"].dropna()
    file_ids = [id for id in df]

    # Create the  vector store and add files.
    vector_store = client.beta.vector_stores.create(name="Handai Assistant V2")
    print(
        f"Created vector store {vector_store.id}. "
        f"Now adding {len(file_ids)} files. This might take few min..."
    )
    start_time = datetime.datetime.now()
    batch = client.beta.vector_stores.file_batches.create_and_poll(
        vector_store_id=vector_store.id, file_ids=file_ids
    )
    print(
        f"  added {len(file_ids)} files with batch id {batch.id}"
        f"in {datetime.datetime.now() - start_time}"
    )

    return vector_store.id


def _incrementally_add_files_to_store(client, vector_store_id):
    """If the store already exist, we can incrementally add files to it."""
    input_references_path = "/path/to/new/files.csv"

    df = pd.read_csv(input_references_path)["openai_file_id"].dropna()
    file_ids = [id for id in df]

    start_time = datetime.datetime.now()
    batch = client.beta.vector_stores.file_batches.create_and_poll(
        vector_store_id=vector_store_id, file_ids=file_ids
    )
    print(
        f"  added {len(file_ids)} files with batch id {batch.id}"
        f"in {datetime.datetime.now() - start_time}"
    )


OPENAI_CLIENT = OpenAI()
IS_FEW_SHOT = False

# If you already have the vector store setup, you can just use the old one.
# It's okay to reveal the id publically because you must be authenticated
# to actually use it.
vector_store_id = "vs_jbxCrdb80BZBWUIteKMnLO6u"
# vector_store_id = _create_vector_store(OPENAI_CLIENT)
# _incrementally_add_files_to_store(OPENAI_CLIENT, vector_store_id)

if IS_FEW_SHOT:
    assistant = OPENAI_CLIENT.beta.assistants.create(
        name="HandAI Assistant V2 (few shot)",
        instructions=create_instructions_for_assistant(),
        model="gpt-4o",
        tools=[{"type": "file_search"}],
        tool_resources={"file_search": {"vector_store_ids": [vector_store_id]}},
    )
else:
    assistant = OPENAI_CLIENT.beta.assistants.create(
        name="HandAI Assistant V2 (zero shot)",
        model="gpt-4o",
        tools=[{"type": "file_search"}],
        tool_resources={"file_search": {"vector_store_ids": [vector_store_id]}},
    )

print(f"created asssistant with id {assistant.id}")

print("done :)")
