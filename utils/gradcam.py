import numpy as np
import cv2
import tensorflow as tf

def get_gradcam_heatmap(model, image, last_conv_layer_name):
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model([image])
        loss = predictions[:, 0]

    grads = tape.gradient(loss, conv_outputs)

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # Convert to numpy safely
    heatmap = heatmap.numpy()

    # ReLU
    heatmap = np.maximum(heatmap, 0)

    # Normalize
    max_val = np.max(heatmap)
    if max_val == 0:
        return np.zeros_like(heatmap)

    heatmap = heatmap / max_val

    return heatmap

def overlay_heatmap(heatmap, image_path, alpha=0.4):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (224, 224))

    heatmap = cv2.resize(heatmap, (224, 224))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    superimposed_img = heatmap * alpha + img

    return np.uint8(superimposed_img)