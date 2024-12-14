![Roboflow](https://img.shields.io/badge/roboflow-6706CE?style=for-the-badge&logo=roboflow&logoColor=#6706CE)
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Git](https://img.shields.io/badge/git-%23F05033.svg?style=for-the-badge&logo=git&logoColor=white)
![GitHub](https://img.shields.io/badge/github-%23121011.svg?style=for-the-badge&logo=github&logoColor=white)

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

## 1.1.1 Getting images from webcam

Assuming you have a webcam on the machine you're working on, we're going to pull images from that webcam upon request.

First let's get openCV to start communicating with our desired camera
```angular2html
# Change the param from 0 up to num of video devices connected
video = cv2.VideoCapture(0)
```

[Note that if you're running **mac on sonoma**, your webcam is currently **unsupported**, but you can try [this cool feature](#bonus) for a potential workaround]

We should also correctly release these resources at the bottom of our script
```angular2html
# Release resources when finished
video.release()
cv2.destroyAllWindows()
```
## Main loop
We want our program to take in a frame from our camera, process it, display it and loop until we decide we're bored (or until you've finished filming yourself and your mates in little boxes to plaster over social media).

So we'll make a loop (I wish I didn't need to specify but this goes before we release the camera).
We'll loop until you press "q":

```angular2html
while 1:
    # On "q" keypress, exit
    if cv2.waitKey(1) == ord('q'):
        break
```

Now, we want to process our image an then display it with (if any exist) predictions returned from our model:

```angular2html
# Main loop; infers sequentially until you press "q"
while 1:
    # On "q" keypress, exit
    if cv2.waitKey(1) == ord('q'):
        break

    # Synchronously get a prediction from the Roboflow API
    image = infer()

    # And display the inference results
    cv2.imshow('image', image)
```
or
```angular2html
# Main loop; infers sequentially until you press "q"
while cv2.waitKey(1) != ord('q'):
    # Synchronously get a prediction from the Roboflow API
    image = infer()

    # And display the inference results
    cv2.imshow('image', image)
```
I'll let you decide what's best

### Process our image
Let's implement our ```infer() function```. We'll get the current frame and return from new frame:
```angular2html
def infer():
    # Get the current frame from the webcam
    _, img = video.read()

    return new_image
```

We could now send our image straight to our model, but that's not a great idea for a couple of reasons:
- Our model was trained with images only of size 416x416 and, in fact, that's the input it's expecting
  - If we decide we want to give it an image of different size we'll also certainly get an unexpected response if it isn't a flat out error
- Our model isn't actually expecting an image, at least in the way you're probably thinking
  - What is really wants is a base 64 string that represents the image
  - We could send the data in a different format inside the request but we'd probably send more data than necessary just for the image to be converted once it arrives anyway
- So let's just both locally and then we'll call our model

First, let's change our image to 416 x 416 whilst still **maintaining the aspect ratio**. This is important as our model has learnt off of normal human faces, not distorted ones!
```angular2html
# Resize (while maintaining the aspect ratio) to improve speed and save bandwidth
height, width, _ = img.shape
scale = 416 / max(height, width)
img = cv2.resize(img, (round(scale * width), round(scale * height)))
```

and now, we can encode it too:
```angular2html
# Encode image to base64 string
_, buffer = cv2.imencode('.jpg', img)
img_str = base64.b64encode(buffer)
```

These are all pretty standard operations, but if you wanted to ask about anything Oscar and I are there in person for a reason!

## Using our model
Now our image is ready to be sent off and processed by the model. How are we going to do this? [Roboflow](https://roboflow.com/) is great for computer vision projects and it provides a great suite to train a network. Given that training a network takes hours, even if we exclude the time to find, augment, preprocess a dataset, tune hyper parameters etc, I've got a pretrained model for you to use. [For now!](#your-own-model)

```angular2html
# Get prediction from Roboflow Infer API
resp = requests.post(UPLOAD_URL, data=img_str, headers={
    "Content-Type": "application/x-www-form-urlencoded"
}, stream=True,timeout=5.0).raw
```

## Your own model

## Bonus

If you're on a mac, you can change the ```video = cv2.VideoCapture(0)``` to 1 or 2 and it should connect to your phone camera which is cool :)





