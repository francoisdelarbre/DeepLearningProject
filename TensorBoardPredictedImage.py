"""class that puts prediction images in tensorboard inspired from
https://stackoverflow.com/questions/43784921/how-to-display-custom-images-in-tensorboard-using-keras?rq=1"""
from PIL import Image
import io
import tensorflow as tf
import numpy as np
from tensorflow.keras.callbacks import Callback


def make_image(tensor):
    """
    Convert an numpy representation image to Image protobuf.
    Copied from https://github.com/lanpa/tensorboard-pytorch/
    """

    height, width, channel = tensor.shape
    if channel == 1:
        tensor = tensor[:, :, 0]
    image = Image.fromarray((tensor*255).astype('uint8'))
    output = io.BytesIO()
    image.save(output, format='PNG')
    image_string = output.getvalue()
    output.close()
    return tf.Summary.Image(height=height,
                            width=width,
                            colorspace=channel,
                            encoded_image_string=image_string)


class TensorBoardPredictedImage(Callback):
    """class used to save the predictions of the model at different epochs to get an idea of its performances"""
    def __init__(self, img, label, model, log_dir):
        super().__init__()
        self.model = model
        self.log_dir = log_dir
        self.input = img
        self._log_img(label, 'label', 0)

    def on_epoch_end(self, epoch, logs={}):
        if epoch % 10 == 0:
            prediction = self.model.predict(self.input)
            prediction = np.round(prediction)  # threshold at 0.5
            self._log_img(prediction, 'prediction', epoch)

    def _log_img(self, img, img_name, epoch):
        image = make_image(img[0, :, :, :])  # getting rid of dimension "batch_size"
        summary = tf.Summary(value=[tf.Summary.Value(tag=img_name, image=image)])
        writer = tf.summary.FileWriter(self.log_dir)
        writer.add_summary(summary, epoch)
        writer.close()
