import cv2
import os
import tqdm
###***Apply for single image***
# def apply_clahe(img_path, clip_limit=0.5, tile_size=(8,8)):
#     # Load grayscale image
#     img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

#     # Apply CLAHE
#     clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
#     clahe_img = clahe.apply(img)

#     return clahe_img

# # Apply CLAHE to image.jpg with default parameters
# clahe_img = apply_clahe('AMRD26.tiff')

# # # Display original and CLAHE images
# cv2.imshow('Original', cv2.imread('AMRD26.tiff', cv2.IMREAD_GRAYSCALE))
# cv2.imshow('CLAHE', clahe_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# Define the path of the input and output folders
img_dir = "C:\\Users\\mkhan\\Desktop\\musabi\\OCT_project\\data\\oct\\ICPSR\\ARMD"
#img_dir = "C:\\Users\\mkhan\\Desktop\\model_testing\\Dubai_751\\africa_study\\dubai_data_234\\"
out_dir = "C:\\Users\\mkhan\\Desktop\\musabi\\OCT_project\\ARMD\\"

#out_dir = "C:\\Users\\mkhan\\Desktop\\model_testing\\Dubai_751\\africa_study\\dubai_data_234_clahe\\"

# Create the output folder if it doesn't exist
# if not os.path.exists(output_folder):
#     os.makedirs(output_folder)

### Function to apply CLAHE 
def apply_clahe(img_path, clip_limit=0.5, tile_size=(8,8)):
    # Load grayscale image
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    # Apply CLAHE
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
    clahe_img = clahe.apply(img)

    return clahe_img

# Loop through all images in the input folder
for filename in os.listdir(img_dir):
    # Read the image
    img = cv2.imread(os.path.join(img_dir, filename))
    print(filename)
    print(img.shape)
    
    equalized = apply_clahe(img)
    
    # Save the processed image to the output folder
    cv2.imwrite(os.path.join(out_dir, filename), equalized)
    
    # Print a message to show progress
    #print(f"Processed {filename}")