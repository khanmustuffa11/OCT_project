import cv2
import os

def apply_clahe_folder(input_folder, output_folder, clip_limit=2.0, tile_size=(8,8)):
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Loop through all files in input folder
    for filename in os.listdir(input_folder):
        if filename.endswith('.jpeg') or filename.endswith('.png') or filename.endswith('.tiff') or filename.endswith('.jpg'):  # Only process image files
            # Load grayscale image
            input_path = os.path.join(input_folder, filename)
            img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)

            # Apply CLAHE
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
            clahe_img = clahe.apply(img)

            # Save CLAHE image in output folder
            output_path = os.path.join(output_folder, 'clahe_' + filename)
            cv2.imwrite(output_path, clahe_img)


# Apply CLAHE to all grayscale images in input folder and save in output folder with default parameters
apply_clahe_folder('C:\\Users\\mkhan\\Desktop\\musabi\\OCT_project\\data\\oct\\oct_artelus\\val\\7', 'C:\\Users\\mkhan\\Desktop\\musabi\\OCT_project\\data\\oct\\oct_artelus\\val_clahe\\7\\')

# Apply CLAHE to all grayscale images in input folder and save in output folder with custom parameters
# apply_clahe_folder('input_folder', 'output_folder', clip_limit=4.0, tile_size=(16,16))

