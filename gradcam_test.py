import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.cm as cm
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Model, Sequential
import os
from cv2 import resize, hconcat,cvtColor,COLOR_GRAY2BGR,vconcat
import numpy as np
model = load_model('C:/Users/mkhan/Desktop/musabi/OCT_project/oct_mendely_artelus_models/model_artmen.h5')
#model = load_model('oct_models/oct_model.h5')
model.layers[-1].activation = None
for layer in model.layers:
    print(layer.name)

def show_image_mask(path):
    layer_input = model.get_layer('input_1').input
    extraction_model = Model(inputs=layer_input, outputs=model.get_layer('multiply').output)
    img = keras.preprocessing.image.load_img(img_path,target_size=(500,500))
    img_array = keras.preprocessing.image.img_to_array(img)
    im = np.array([img_array])
    w = extraction_model.predict(im)
    im_res = np.squeeze(im)[:,:,0]
    im_w = np.power(resize(np.median(w[0], axis=2), (500, 500)),4)
    out_img = np.multiply(im_w, im_res) * 255
    out_img = cvtColor(out_img.astype('uint8'), COLOR_GRAY2BGR)
    return out_img+im_input
    

def make_gradcam_heatmap(img_path, model, last_conv_layer_name, pred_index=None):
    # First, we create a model that maps the input image to the activations
    # of the last conv layer as well as the output predictions
    img = keras.preprocessing.image.load_img(img_path,target_size=(500,500))
    img_array = keras.preprocessing.image.img_to_array(img)
    im = np.array([img_array])
    print(model)
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(im)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # This is the gradient of the output neuron (top predicted or chosen)
    # with regard to the output feature map of the last conv layer
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    # then sum all the channels to obtain the heatmap class activation
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def save_and_display_gradcam(img_path, heatmap_conv, heatmap_multiply, alpha=.4):
    # Load the original image
    img = keras.preprocessing.image.load_img(img_path,target_size=(500,500))
    img_array = keras.preprocessing.image.img_to_array(img)
    out_img = img_array
    for heatmap in [heatmap_conv, heatmap_multiply]:
        # Rescale heatmap to a range 0-255
        heatmap = np.uint8(255 * heatmap)

        # Use jet colormap to colorize heatmap
        jet = cm.get_cmap("jet")

        # Use RGB values of the colormap
        jet_colors = jet(np.arange(256))[:, :3]
        jet_heatmap = jet_colors[heatmap]

        # Create an image with RGB colorized heatmap
        jet_heatmap = keras.preprocessing.image.array_to_img(jet_heatmap)
        jet_heatmap = jet_heatmap.resize((img_array.shape[1], img_array.shape[0]))
        jet_heatmap = keras.preprocessing.image.img_to_array(jet_heatmap)
        # Superimpose the heatmap on original image
        superimposed_img = jet_heatmap * alpha + img_array
        out_img = hconcat([out_img, superimposed_img])
        
    superimposed_img = keras.preprocessing.image.array_to_img(out_img)
    
    cam_path = 'gradcam_'+os.path.basename(img_path)
    # Save the superimposed image
    superimposed_img.save(cam_path)
   
for f in ['data/oct/oct_100/1/DRUSEN-11129-3.jpeg','data/oct/oct_100/1/DRUSEN-186682-2.jpeg']:
    img_path = f
    heatmap_conv = make_gradcam_heatmap(img_path, model, 'efficientnetv2-b0')
    heatmap_multiply = make_gradcam_heatmap(img_path, model, 'efficientnetv2-b0')
    save_and_display_gradcam(img_path, heatmap_conv, heatmap_multiply) 