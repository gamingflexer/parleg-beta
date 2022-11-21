import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random

# import some common detectron2 utilities
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer, ColorMode
import matplotlib.pyplot as plt
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.data import MetadataCatalog

import sys
from utils import data_to_csv

# add the custom path 
sys.path.insert(0, os.path.abspath('./detectron2'))

register_coco_instances("my_dataset_train2", {}, "/content/train.json", "/content/images")
register_coco_instances("my_dataset_val2", {}, "/content/test.json", "/content/images")
dataset_metadata = MetadataCatalog.get("my_dataset_train2")
print("Dataset Loaded\n")

'''gdown 14J0lKNYS3LLcDLZ-lkugYs9MkvruVavr for saved model'''

#download trained model # Downloading save model and using can lead to bad results beacuse of LACK of dataset #skip this cell if you trined model which takes 2 min only so prefer that
# !unzip /content/detercon2_custom_model.zip

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.WEIGHTS = "/content/MODELS_TRAINED/model_final.pth"  # add model path here
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.6
cfg.MODEL.DEVICE = 'cpu' #for cpu else comment this line
predictor = DefaultPredictor(cfg)

# add the images folder where the cropped images are stored particular in black and white
def predictor_custom(img_path,crop=False,bottom=True):
  if crop:
    im = cv2.imread(img_path)
    im0 = cv2.imread(img_path,0)
    height, width = im0.shape
    x = (height/2)
    if bottom == True:
      im2 = im[int(x):]
    else:
      im2 = im[:int(x)]
    outputs = predictor(im2)
    v = Visualizer(im2[:, :, ::-1],
                    metadata=dataset_metadata,
                    scale=0.5,
                    instance_mode=ColorMode.IMAGE_BW
                    )
  else:
    im = cv2.imread(img_path)
    outputs = predictor(im)
    v = Visualizer(im[:, :, ::-1],
                      metadata=dataset_metadata,
                      scale=0.5,
                      instance_mode=ColorMode.IMAGE_BW
                      )
  out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
  img = cv2.cvtColor(out.get_image()[:, :, ::-1], cv2.COLOR_RGBA2RGB)
  #cv2_imshow(img) use this if you are using colab
  return outputs["instances"].to("cpu"),outputs["instances"].pred_boxes,outputs["instances"].pred_classes

'''RUN HERE'''

#add the path of the images folder
df = data_to_csv("/content/images")