"""
Helper functions to create folders in google drive
"""

from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from google.auth.transport.requests import Request
import os.path
import pickle
from dataclasses import dataclass
from google.auth.exceptions import RefreshError


@dataclass
class DriveDirectory:
    """Created directory"""

    question_num: str
    reference_num: str
    folder_id: str


# The scope determines what you can access. This scope allows for viewing files in Google Drive.
SCOPES = ["https://www.googleapis.com/auth/drive"]


def get_creds():
    creds = None
    # The file token.pickle stores the user's access and refresh tokens. It is
    # created automatically when the authorization flow completes for the first time.
    if os.path.exists("token.pickle"):
        with open("token.pickle", "rb") as token:
            creds = pickle.load(token)
            print(f"Token loaded. Valid: {creds.valid}")

    # If there are no (valid) credentials available, let the user log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            try:
                creds.refresh(Request())
                print("Token refreshed successfully.")
            except RefreshError:
                print("Token refresh failed. Need to re-authenticate.")
                creds = None
        if not creds:
            flow = InstalledAppFlow.from_client_secrets_file(
                "client_secret.json", SCOPES
            )
            creds = flow.run_local_server(port=0)
            print("Authenticated and received new credentials.")
        # Save the credentials for the next run
        with open("token.pickle", "wb") as token:
            pickle.dump(creds, token)
            print("Token saved to token.pickle.")

    return creds


def create_folder(service, name, parent_id=None):
    folder_metadata = {
        "name": name,
        "mimeType": "application/vnd.google-apps.folder",
    }
    if parent_id:
        folder_metadata["parents"] = [parent_id]
    folder = service.files().create(body=folder_metadata, fields="id").execute()
    return folder.get("id")


def write_directories(
    structure: "dic[str,list[str]]", root_folder_name: str
) -> "list[DriveDirectory]":
    """
    Create directory structure in drive
    input:
        structure: how to structure directory. E.g. {1: [1, 2], 24: [1], 34: [1, 2, 3]}
            will create the following structure
                /root
                /root/question_1
                /root/question_1/reference_1
                /root/question_1/reference_2
                /root/question_24
                /root/question_24/reference_1
                /root/question_34
                /root/question_34/reference_1
                /root/question_34/reference_2
                /root/question_34/reference_3
    """
    ret = []

    print("creating drive folders for references...")
    creds = get_creds()
    service = build("drive", "v3", credentials=creds)
    # Create the top-level folder
    root_folder_id = create_folder(service, root_folder_name)
    print(f"created /{root_folder_name}")

    # Create a nested folders
    for id, ref_nums in structure.items():
        question_folder_name = f"question_{id}"
        nested_folder_id = create_folder(
            service, question_folder_name, root_folder_id
        )
        print(f"created /{root_folder_name}/{question_folder_name}")
        for n in ref_nums:
            reference_folder_name = f"reference_{n}"
            folder_id = create_folder(
                service, reference_folder_name, nested_folder_id
            )
            print(
                f"created /{root_folder_name}/{question_folder_name}/{reference_folder_name}"
            )
            ret.append(
                DriveDirectory(
                    question_num=id, reference_num=n, folder_id=folder_id
                )
            )

    return ret


def main():
    creds = get_creds()
    service = build("drive", "v3", credentials=creds)
    # Create the top-level folder
    top_folder_id = create_folder(service, "TopLevelFolder")

    # Create a nested folder
    nested_folder_id = create_folder(service, "NestedFolder1", top_folder_id)

    print(f"Top Folder ID: {top_folder_id}")
    print(f"Nested Folder 1 ID: {nested_folder_id}")


if __name__ == "__main__":
    main()
