import csv
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from tensorflow import keras

import loss
from utils import get_run_length_enc, predict_image

# some versions of keras cannot load a model with custom losses.
# see https://github.com/keras-team/keras/issues/5916
custom_losses = loss.__dict__


def make_submission_file(model=None, model_dir=None, model_name="model.h5", images_dir="data/stage2_test_final"):
    log_dir_path = Path("tf_logs")
    images_dir = Path(images_dir)

    log_dir_paths = sorted(log_dir_path.glob("*"), reverse=True)

    if model_dir is None:
        for log_dir_path in log_dir_paths:
            if (log_dir_path / "model.h5").exists():
                model_dir = log_dir_path
                print("using model directory %s" % log_dir_path.name)
                break
    else:
        model_dir = Path(model_dir)

    model_path = model_dir / model_name

    submission_dir = (model_dir / "submissions")
    submission_dir.mkdir(exist_ok=True)
    submission_path = submission_dir / (datetime.now().strftime("%Y.%m.%d-%H.%M") + ".csv")

    if model_path is None:
        raise ValueError("{} does not contain any valid log dir".format(str(log_dir_path)))

    if model is None:
        model = keras.models.load_model(model_path, custom_objects=custom_losses)

    images_ids = [image_dir.name for image_dir in images_dir.iterdir() if image_dir.is_dir()]
    images_paths = (images_dir / image_id / "images" / (image_id + ".png")
                    for image_id in images_ids)

    images = (cv2.imread(str(image_path)) / 255. for image_path in images_paths)

    predictions = (predict_image(model, image) for image in images)

    with open(str(submission_path), 'w') as submission_file:
        csv_writer = csv.writer(submission_file)
        csv_writer.writerow(["ImageId", "EncodedPixels"])

        for image_id, prediction in zip(images_ids, predictions):
            nb_label, mask = cv2.connectedComponents((prediction * 255).astype(np.uint8))
            if nb_label > 1:
                for label in range(nb_label):
                    if label:
                        current_prediction = np.zeros(prediction.shape, dtype=np.float32)
                        current_prediction[mask == label] = 1.
                        prediction_enc = get_run_length_enc(current_prediction)
                        csv_writer.writerow([image_id, " ".join((str(elem)
                                                                 for elem in prediction_enc))])
            else:
                csv_writer.writerow([image_id, ""])

    return model
