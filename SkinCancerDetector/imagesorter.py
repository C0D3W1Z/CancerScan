import os
import csv
import shutil

# Path to the directory containing the images
images_dir = "./Skin Cancer HAM10000/images"

# Path to the CSV file
csv_path = "./Skin Cancer HAM10000/HAM10000_metadata.csv"

# Read the CSV file
with open(csv_path, "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        # Get the image ID
        image_id = row["image_id"]

        # Get the image file path
        image_path = os.path.join(images_dir, image_id + ".jpg")

        # Get the type value from the CSV
        image_type = row["dx"]

        # Create a folder for the image type if it doesn't exist
        folder_path = os.path.join(".", image_type)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # Copy the image file to the folder for its type
        shutil.copy(image_path, os.path.join(folder_path, image_id + ".jpg"))
