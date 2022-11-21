import pandas as pd
import numpy as np
from ocr import ocr
from main import predictor_custom

import os

classes_main = {0:"a",1:"b",2:"c",3:"d",4:"e",5:"f",6:"g",7:"h",8:"i"}
orignal_classes = {"a":1,"b":2,"c":3,"d":4,"e":5,"f":6,"g":7,"h":8,"i":9}

def output_corrector(boxes,classes):
  boxes_array, classes_array = boxes.tensor.cpu().numpy(),classes.cpu().numpy()
  symbols_found = ""
  if len(boxes_array) != 0:
    len_bbox_found = len(boxes_array)
    contours = []
    if len_bbox_found >0:
      for i in range(0,len_bbox_found):
        contours.append(tuple(boxes_array[i]))
    else:
      print("Nothing found")

    arr = np.array(contours)
    r = (arr[:, 0]**2 + arr[:, 1] **2)**0.5
    for m in np.argsort(r):
      for b,c in zip(boxes_array,classes_array):
        tuple_bbox = tuple(b)
        if tuple(boxes_array[m]) == tuple_bbox:
          symbols_found = symbols_found + classes_main[c]
  return symbols_found

def data_to_csv(folder_path):
  symbols = ""
  df = pd.DataFrame({}, columns=['Device Name','REF','LOT','Qty','Symbols'])
  os_dir = os.listdir(folder_path)
  for i in os_dir:
    image_path_one = folder_path + i
    try:
      ocr_data = ocr(image_path_one)
      print("OCR done for.......",i)
      #ocr
      for data_one in ocr_data:
        if data_one.isdigit():
          qty = data_one
        if "Device" in data_one:
          device_Name = data_one.split(":")[1]
        if "LOT" in data_one:
          lot = data_one.split(":")[1]
        if "REF" in data_one:
          ref = data_one.split(" ")[1]

      #for the top part 
      out, pred_boxes, classes = predictor_custom(image_path_one,crop=True,bottom=False)
      symbols_top = output_corrector(pred_boxes, classes)

      #for the bottom part
      out, pred_boxes, classes = predictor_custom(image_path_one,crop=True,bottom=True)
      symbols_bottom = output_corrector(pred_boxes, classes)

      symbols_all = symbols_top + symbols_bottom
      symbols_found_list = list(symbols_all)
      final_symbols = ""
      for sym in symbols_found_list:
        final_symbols = final_symbols + str(orignal_classes[sym])
        
      df.loc[len(df.index)] = [device_Name,ref,lot,qty,final_symbols]
      print("detection done for.......",i,"---- ",final_symbols)
    except ValueError:
      print("for files .ipynb_checkpoints")
      pass
  return df