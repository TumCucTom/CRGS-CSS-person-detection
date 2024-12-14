# CRGS-CSS-person-detection

## 0.1 Download this repo
Download this repo via the zip (and unzip it) or with:
```angular2html
git clone git@github.com:TumCucTom/CRGS-CSS-person-detection.git
```
or
```angular2html
git clone github.com/TumCucTom/CRGS-CSS-person-detection/
```
If you don't have git setup on your machine:
- You should - but do this later
- Just download the .zip for now and live in shame as your classmates mock you

You can now open the project in your favourite IDE

## 0.2  Setup your machine

Download [python 3](https://www.python.org/downloads/) if you have not already.

If your terminal does not say ```CRGS-CSS-person-detection %``` or similar, go into this project which your just downloaded:
```angular2html
cd CRGS-CSS-person-detection
```

Setup your python environment with
```
python3 -m venv [path to your env]
source [path to your env]/bin/activate
```

Your console should now look something like this ```(env) You@Your-machine-name ~ %```. Slighty different if you are on windows (eugh). What's important is that you are now in your environment.

Now install the dependencies needed for the project with:
```angular2html
pip3 install requirements.txt
```
## 1.1 Using a prebuilt NN

Let's start coding now!

Open the empty python file in the root of this directory. Probably by double clicking on the file "person-detection.py" in your IDE. Or if you're crazy:
```angular2html
nano person-detection.py
vim person-detection.py
```

First let's import the dependencies:

- We want base64 for encoding our image
- OpenCV (cv2) for taking in and outputting images
- Numpy for performing some calculations to get our image to the right input size for our NN model
- Requests so that we can make RESTful calls to roboflow
- Because we're not cavemen, we'll also put a docstring too

```angular2html
"""Put bounding boxes around people in a room"""
import base64
import cv2
import numpy as np
import requests
```






