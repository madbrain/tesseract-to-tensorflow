
# Tesseract to Tensorflow model converter

This software is a simple python implementation of the [Tesseract](https://github.com/tesseract-ocr/tesseract) recognizer.

It can:
* read the Tesseract data model from `eng.traineddata` file
* recognize words from an input image containing a text line
* convert/extract the network model coefficients to a format suitable to be run by a tensorflow model

# Installation

```
apt get install tesseract-ocr-eng
pip3 install tensorflow (2.13.1)
pip3 install numpy==1.23
```

# Running

```
python3 -m tesseract.convert

cd tensorflow
python3 run.py
```

# TODO
* use [deepdream](https://www.tensorflow.org/tutorials/generative/deepdream) technics to give an interpretation to the architecture and the coefficients