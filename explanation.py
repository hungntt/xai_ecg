import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from dataloader.ptbdb_loader import dataloader


def gradcam(model, x, index, layer_name):
    # Convert input to tensor
    x = tf.convert_to_tensor(x, dtype=tf.float32)

    # Create an intermediate model that outputs the desired layer's output and the final predictions
    intermediate_model = tf.keras.models.Model(inputs=model.inputs,
                                               outputs=[model.get_layer(layer_name).output, model.output])

    with tf.GradientTape() as tape:
        # Watch the input tensor for gradient computation
        tape.watch(x)

        # Forward pass through the intermediate model to get both layer outputs and predictions
        layer_output, predictions = intermediate_model(x)

        # Extract the prediction index and compute the loss
        pred_index = tf.argmax(predictions[index])
        loss = predictions[:, pred_index]

    # Compute gradients of the target class (indexed by pred_index) with respect to the layer_output
    grads = tape.gradient(loss, layer_output)

    # Pool the gradients across the width of the input
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1))

    # Use the pooled gradients and the layer's output to compute the heatmap
    heatmap = tf.reduce_mean(tf.multiply(pooled_grads, layer_output), axis=-1)

    # Normalize the heatmap
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    heatmap = heatmap.numpy()  # Convert to numpy array for visualization

    return heatmap, pred_index.numpy()


def heatmap_(heatmap, data, title):
    heatmap_2d = np.expand_dims(heatmap, axis=1)
    heatmap_2d = cv2.resize(heatmap_2d, (100, 187))

    figure = plt.figure(figsize=(80, 4))
    ax = figure.add_subplot(111)
    x = np.arange(0, 187, 1)
    ax.plot(x, data * 100)
    ax.imshow(np.transpose(heatmap_2d), cmap="Reds", extent=[0, 187, -10, 120])
    ax.autoscale(False)
    ax.set_xlim(0, 187)
    ax.set_ylim(-10, 110)
    ax.set_title(title)
    plt.show()


def heatmap_3x(data, name_list, Collect_heatmap):
    def preprocess_heatmap(heatmap):
        # Assuming heatmap is a 1D array of shape (187,)
        heatmap = np.expand_dims(heatmap, axis=0)  # Now it has shape (1, 187)
        heatmap_2d = cv2.resize(heatmap, (187, 100))  # Resize to (100, 187) for display

        # Ensure the output is 2D by explicitly selecting the first channel if necessary
        # This step assumes the output might incorrectly have a third dimension
        if heatmap_2d.ndim == 3:
            heatmap_2d = heatmap_2d[:, :, 0]  # Correcting to ensure 2D shape

        return heatmap_2d

    x = np.arange(0, 187, 1)

    fig = make_subplots(rows=1, cols=len(name_list))
    for i, name in enumerate(name_list, start=1):
        # Add ECG line plot for each subplot
        fig.add_trace(go.Scatter(x=x, y=data * 100, name="ECG", line=dict(color="black")), row=1, col=i)

        # Process heatmap data
        heatmap_2d = preprocess_heatmap(Collect_heatmap[i - 2])  # Adjust your preprocessing here
        heatmap_2d[heatmap_2d < np.percentile(heatmap_2d, 90)] = 0
        heatmap_2d = cv2.GaussianBlur(heatmap_2d, (5, 5), 0)
        heatmap_2d = (heatmap_2d - np.min(heatmap_2d)) / (np.max(heatmap_2d) - np.min(heatmap_2d))

        # Since Plotly doesn't overlay line plots and heatmaps directly, we simulate this by adding a heatmap
        fig.add_trace(go.Heatmap(z=heatmap_2d, colorscale="Purples", showscale=False), row=1, col=i)

    return fig


def explain(index):
    index = int(index)
    X, _ = dataloader()  # Assuming dataloader is accessible and returns the full dataset
    selected_X = X.iloc[index]
    model = tf.keras.models.load_model('resnet1d.h5')
    pred = model.predict(selected_X.values.reshape(1, 187, 1))
    pred = dict(zip(["HC", "MI"], pred[0]))
    Collect_heatmap = list()
    name_list = ["block4_conv1"]
    for name in name_list:
        heatmap, pred_class = gradcam(model, X, index, name)
        Collect_heatmap.append(heatmap)
    fig = heatmap_3x(selected_X, name_list, Collect_heatmap)
    return fig, pred  # Format the prediction as needed
