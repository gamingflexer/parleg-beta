import easyocr
import cv2

text_reader = easyocr.Reader(['en']) #Initialzing the ocr

def ocr(path):
  img = cv2.imread(path, 0) 
  results = text_reader.readtext(img)
  data = []
  for (bbox, text, prob) in results:
      data = data + [text]
  return data