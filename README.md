# Parleg Beta | OCR and Logo Detection & Recognition

<img src="https://img.shields.io/static/v1?label=Detectron2&message=Used&color=PURPLE"/>

[Collab Link](https://colab.research.google.com/drive/1W3C6nRJUkuT_Tdpd6Z73yAOzLvzLcRZJ?usp=sharing)

As we have done in the [Parleg-Alpha](https://github.com/gamingflexer/parleg-alpha) project, We have successfully implemented the croping of the required part of the image.
The project is divided into two parts because we get video footage of the boxes moving on a conveyor belt. In this project, we have implemented the OCR and Logo Detection & Recognition in a particular order. 

## Problems to solve 

- OCR of the image
- Find and match appropriate matches for REF, LOT, Qty & Device Name.
- Detection of “Symbols” in the images
- Recognition of “Symbols” in the images
- Saving the data in a CSV file in a particular order.

## Solution

- Use easy ocr to extract text from the images and using regular expressions to match the other field required which were REF, LOT, Qty & Device Name
- Annotated images in COCO format
- Augmented dataset as the images were only 8 and very unbalanced. (for ex - number 3 symbols images are very low)
- Trained Faster RCNN Detectron 2 model for detection & recognition of the symbols with custom augmentation classes to do in-training augmentation of images.
- Used the bounding box & simple image processing to find the position of the symbols and add them to excel using pandas dataframe.

One could go by finding contours and making bounding boxes but they generally work good only for text rather than complex objects, instead they can be used as labeling functions.

# Installation

Clone the repository and then install detectron2 using the following command

```
python -m pip install pyyaml==5.1 -q

git clone 'https://github.com/facebookresearch/detectron2'
cd detectron2
pip install -e detectron2
```

# Train the model

In the src folder there is `train.py` read the comments in it and change the paths accordingly and then run the file.
For more information on how to train the model, refer here :

This also contains scripts for dataset conversions and common issues faced while training the model.

- [Easy to Train Detectron2](https://github.com/gamingflexer/object-detection-custom-models-scripts)

- [Detectron2 Documentation](https://detectron2.readthedocs.io/en/latest/tutorials/datasets.html)

# Test the model

Run `main.py` in the src folder and change the paths accordingly for the folder to be tested.