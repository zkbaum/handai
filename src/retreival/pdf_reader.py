"""
Pre-process PDFs by converting them to text files. This is necessary for the 
chatcompletions version of retreival, where we inject the articles as text into
the prompt.
"""

import os
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import io
from private import ROOT_DIR


def read_pdf_with_ocr(pdf_path, tesseract_cmd=None):
    """
    Reads a PDF file and extracts text from images using OCR.

    Parameters:
    - pdf_path: str, path to the PDF file to be processed.
    - tesseract_cmd: str, optional, path to the Tesseract-OCR executable.

    Returns:
    - A dictionary with page numbers as keys and extracted text from images as values.
    """
    print(f"  using ocr to read {pdf_path}")
    # Configure Tesseract-OCR path if provided
    if tesseract_cmd:
        pytesseract.pytesseract.tesseract_cmd = tesseract_cmd

    # Open the PDF file
    pdf = fitz.open(pdf_path)

    # Initialize a dictionary to hold extracted text
    extracted_text = {}

    # Iterate through each page of the PDF
    for page_num in range(len(pdf)):
        page_text = []  # List to hold text of images from the current page

        # Get the page
        page = pdf.load_page(page_num)

        # Get the images in the page
        for image_index, img in enumerate(page.get_images(full=True)):
            # Extract the image using its XREF
            xref = img[0]
            base_image = pdf.extract_image(xref)
            image_bytes = base_image["image"]

            # Convert the bytes to a PIL Image
            image = Image.open(io.BytesIO(image_bytes))

            # Use pytesseract to do OCR on the image
            text = pytesseract.image_to_string(image)

            # Add the extracted text to the page text list
            page_text.append(text)

        # Combine text from all images into a single string for the current page
        extracted_text[page_num] = "\n".join(page_text)

    # Close the PDF after processing
    pdf.close()

    print(f"  finished reading {pdf_path} with ocr")

    return extracted_text


def read_pdf(file_path):
    """
    Read the text from a single PDF file.
    """
    text = ""
    with fitz.open(file_path) as doc:
        for page in doc:
            text += page.get_text()
    return text


def write_text_for_pdfs(start_path):
    """
    Traverse through the references directory and convert all PDFs
    into text files.
    """
    for root, dirs, files in os.walk(start_path):
        for file in files:
            if file.endswith(".pdf"):
                pdf_path = os.path.join(root, file)
                print(f"Reading {pdf_path}...")

                pdf_text = read_pdf(pdf_path)
                if len(pdf_text) == 0:
                    pdf_dict = read_pdf_with_ocr(pdf_path)
                    pdf_text = "".join(pdf_dict.values())

                text_file_path = os.path.join(
                    root, file.replace(".pdf", "_processed.txt")
                )
                with open(text_file_path, "w") as text_file:
                    text_file.write(pdf_text)
                print(f"  wrote text to {text_file_path}")


# write_text_for_pdfs(f"{ROOT_DIR}/data/references/handai-2013-references/drive")
write_text_for_pdfs(f"{ROOT_DIR}/data/references/handai-2012-references/drive")
