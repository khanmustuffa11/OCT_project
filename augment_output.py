import tensorflow as tf
import tensorflow_addons as tfa
import random
import math

def __data_augmentation(img, mode='rgb'):
    flip_prob = tf.random.uniform([], 0, 1.0, dtype=tf.float32)
    if flip_prob > 0.5:
        img = tf.image.transpose(img)
    img = tf.image.random_flip_up_down(img)
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_brightness(img, 0.5)
    img = tf.image.random_contrast(img, 0.5, 1.5)
    img = tf.image.random_saturation(img, 0.5, 1.5)
    if mode == 'rgb':
        img = tf.image.random_hue(img, 0.5)
    img = tfa.image.rotate(img, random.uniform(-90, 90) * math.pi / 180)
    img = tf.image.random_jpeg_quality(img, 30, 100)

    return img


# Load an example image
img_path = 'NORMAL-114740-5.jpeg'
img = tf.io.read_file(img_path)
img = tf.image.decode_jpeg(img, channels=3)  # Assuming RGB image

# Define the augmentation technique names
techniques = ['transpose', 'flip_up_down', 'flip_left_right', 'brightness', 'contrast', 'saturation', 'hue', 'rotation', 'jpeg_quality']

# Apply data augmentation techniques and save output samples
output_dir = 'augmented_outputs'  # Directory to save augmented images
num_samples = 5  # Number of output samples to generate

# Generate and save output samples
for i in range(num_samples):
    augmented_img = __data_augmentation(img)
    technique_name = techniques[i % len(techniques)]
    output_path = f'{output_dir}/augmented_{technique_name}_{i + 1}.jpg'
    tf.io.write_file(output_path, tf.image.encode_jpeg(augmented_img))

    print(f'Saved augmented image {i + 1} with technique {technique_name} at {output_path}')
