import tensorflow as tf
from tensorflow import keras
from pathlib import Path
from loss import dice_loss, bce_dice_loss, i_o_u_metric
from utils import get_run_length_enc, get_full_resolution, get_resized_images
import cv2
import numpy as np
from matplotlib import pyplot as plt
import csv
from datetime import datetime

# some versions of keras cannot load a model with custom losses. see https://github.com/keras-team/keras/issues/5916
custom_losses = {
    "dice_loss": dice_loss,
    "bce_dice_loss" : bce_dice_loss,
    "i_o_u_metric" : i_o_u_metric
}

# keras.losses.dice_loss = dice_loss
# keras.losses.bce_dice_loss = bce_dice_loss
# keras.losses.i_o_u_metric = i_o_u_metric
def make_submission_file(model_dir=None, model_name="model.h5", images_dir="data/stage1_test"):
    log_dir_path = Path("tf_logs")
    images_dir = Path(images_dir)
    
    log_dir_paths = sorted(log_dir_path.glob("*"), reverse=True)
    
    print(log_dir_paths)
    if model_dir is None:
        for log_dir_path in log_dir_paths:
            if (log_dir_path / "model.h5").exists():
                model_dir = log_dir_path
                print("recovering model from %s" % log_dir_path.name)
                print(model_dir)
                break
    else:
        model_dir = Path(model_dir)
    
    model_path = model_dir / model_name
    
    submission_dir = (model_dir / "submissions")
    submission_dir.mkdir(exist_ok=True)
    submission_path = submission_dir / (datetime.now().strftime("%Y.%m.%d-%H.%M") + ".csv")

    if model_path is None:
        raise ValueError("{} does not contain any valid log dir".format(str(log_dir_path)))

    model = keras.models.load_model(model_path, custom_objects=custom_losses)
    
    images_ids = [image_dir.name for image_dir in images_dir.iterdir() if image_dir.is_dir()]
    images_paths = ( images_dir / image_id / "images" / ( image_id + ".png") 
                   for image_id in images_ids)
    
    images = (cv2.imread(str(image_path)) / 256. for image_path in images_paths)
    
    predictions = (predict_image(model, image) for image in images)
    
    predictions_run_length = list(get_run_length_enc(prediction) for prediction in predictions)
    
    with open(str(submission_path), 'w') as submission_file:
        csv_writer = csv.writer(submission_file)
        csv_writer.writerow(["ImageId", "EncodedPixels"])
        for image_id, prediction_run_length in zip(images_ids, predictions_run_length):
            csv_writer.writerow([image_id, " ".join(str(elem) for elem in prediction_run_length)])
            
        
        

def predict_image(model, image):
    images, cooredonates = get_resized_images(image, model.output.shape[1:3])
    images_array = np.array(images)
    
    predictions = model.predict(images_array)
    

    
    predictions_list = np.split(predictions, predictions.shape[0], axis=0)
    
    for i, prediction in enumerate(predictions_list):
        predictions_list[i] = prediction.squeeze()

        
    
    prediction_full = get_full_resolution(predictions_list, cooredonates, image.shape[:2], round=True)
    
    return prediction_full