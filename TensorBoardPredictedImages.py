"""class that puts prediction images in tensorboard inspired from
https://stackoverflow.com/questions/43784921/how-to-display-custom-images-in-tensorboard-using-keras?rq=1"""
from PIL import Image
import io
import tensorflow as tf
import numpy as np
from tensorflow.keras.callbacks import Callback


def make_image(image, label):
    """
    Convert an numpy representation image to Image protobuf.
    inspired from https://github.com/lanpa/tensorboard-pytorch/
    """

    height, width, channel = image.shape
    if channel == 1:  # adding labels in 3rd dimension
        image_with_label = np.zeros((height, width, 3), dtype=np.uint8)
        image_with_label[:, :, 0] = image[:, :, 0] * 255
        image_with_label[:, :, 2] = label[:, :, 0] * 255
    else:
        raise ValueError('image should be black and white')
    pil_image = Image.fromarray(image_with_label)
    output = io.BytesIO()
    pil_image.save(output, format='PNG')
    image_string = output.getvalue()
    output.close()
    return tf.Summary.Image(height=height,
                            width=width,
                            colorspace=channel,
                            encoded_image_string=image_string)


class TensorBoardPredictedImages(Callback):
    """class used to save the predictions of the model at different epochs to get an idea of its performances, for each
    image, the label is displayed in blue and the prediction in red"""
    def __init__(self, imgs, labels, model, log_dir):
        super().__init__()
        self.model = model
        self.log_dir = log_dir
        self.inputs = imgs
        self.labels = labels

    def on_epoch_end(self, epoch, logs={}):
        if epoch % 10 == 0:
            predictions = self.model.predict(self.inputs)
            sum_value = []
            for i in range(predictions.shape[0]):
                image = make_image(predictions[i, :, :, :], self.labels[i, :, :, :])  # getting rid of dimension
                # "batch_size"
                sum_value.append(tf.Summary.Value(tag='prediction_' + str(i + 1), image=image))

            writer = tf.summary.FileWriter(self.log_dir)
            summary = tf.Summary(value=sum_value)
            writer.add_summary(summary, epoch)
            writer.close()
