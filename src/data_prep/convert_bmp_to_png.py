"""
Some of the exams have images in bmp format, which is not compatible with the 
openAI api. Therefore, we must convert them all to png here. Afterwards, you can
upload to AWS using the S3 console.
"""

from private import ROOT_DIR
from PIL import Image
import os


def convert_bmp_to_png(source_directory, target_directory):
    """
    Convert all BMP images in a source directory to PNG format and
    save them in a target directory with the same subdirectory structure.

    :param source_directory: The directory to search for BMP files.
    :param target_directory: The directory where PNG files will be saved.
    """
    for subdir, dirs, files in os.walk(source_directory):
        for file in files:
            if file.lower().endswith(".bmp"):
                # Construct full file path
                source_file_path = os.path.join(subdir, file)
                # Replace the source directory with target directory and change the file extension to .png
                target_file_path = os.path.join(
                    target_directory,
                    os.path.relpath(subdir, source_directory),
                    file[:-4] + ".png",
                )
                # Ensure target directory exists
                os.makedirs(os.path.dirname(target_file_path), exist_ok=True)
                # Convert the image
                with Image.open(source_file_path) as img:
                    img.save(target_file_path, "PNG")
                    print(f"Converted and saved: {target_file_path}")


# Example usage
# source_directory = 'ROOT_DIR/source/'
# target_directory = 'ROOT_DIR/img'
# convert_bmp_to_png(source_directory, target_directory)
