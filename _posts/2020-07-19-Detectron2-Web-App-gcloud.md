---
layout: post
title:  Low Memory Instance Segmentation
---

Various hacks that helped me deploy an instance segmentation web app to google cloud


Machine Learning models are memory intensive. My [current web app](https://spiyer99.github.io/Detectron2-Web-App/) consumes at least 1GB of memory. That makes it difficult to deploy to the cloud. I kept seeing the dreaded [out of memory error](https://en.wikipedia.org/wiki/Out_of_memory).

The immediate solution that comes to mind is increasing the memory of the VM instance. But I'd rather not spend more money that I have to. This is a side project, after all. 

So I came up with a work around. It's actually inspired by [deoldify](https://github.com/jantic/DeOldify/tree/master/deoldify) by Jason Antic. Jason uses a render factor to downscale the image and convert it to a square before running his machine learning model. 

I thought that applying the same idea here could reduce the memory requirements. 

# The Hack

In short here is the solution. It's larged taken from [here](https://github.com/jantic/DeOldify/blob/master/deoldify/filters.py).

```python
import PIL
from PIL import Image

# taken from: https://github.com/jantic/DeOldify/blob/master/deoldify/filters.py

def _scale_to_square(orig, targ):
  targ_sz = (targ, targ)
  return orig.resize(targ_sz, resample=PIL.Image.BILINEAR)


def _unsquare(image, orig):
  targ_sz = orig.size
  image = image.resize(targ_sz, resample=PIL.Image.BILINEAR)
  return image

```

We scale to square of size `targ`. `targ` is essentially the render factor that Jason mentions in deoldify. 

If the render factor is too high it can [lead to OOM errors](https://github.com/jantic/DeOldify/blob/edac73edf1d3557f95a71f860cffd6c4c91f66f0/deoldify/filters.py#L58). Too low and the result will have a poor resolution. 

What this looks like in practice is pretty simple actually. First we scale the image to a sqaure of size `targ`. Then we run inference on that scaled image. The result from inference is passed through the `_unsquare` function. This converts our square size image to its former size. 

```python
from Detector import Detector
import io
from flask import Flask, render_template, request, send_from_directory, send_file
from PIL import Image
import requests
import os
import img_transforms

app = Flask(__name__)
detector = Detector()

RENDER_FACTOR = 25

# run inference using image transform to reduce memory
def run_inference_transform(img_path = 'file.jpg', transformed_path = 'file_transformed.jpg'):

	# get height, width of image
	original_img = Image.open(img_path)

	# transform to square, using render factor
	transformed_img = img_transforms._scale_to_square(original_img, targ=RENDER_FACTOR*16)
	transformed_img.save(transformed_path)

	# run inference using detectron2
	detector.inference(transformed_path)
	untransformed_result = Image.open('/home/appuser/detectron2_repo/img.jpg')

	# unsquare
	result_img = img_transforms._unsquare(untransformed_result, original_img)

	# clean up
	try:
		os.remove(img_path)
		os.remove(transformed_path)
	except:
		pass

	return result_img
```

After this we can finally deploy on most cloud platforms without any OOM issues. 


# Deployment on Google Cloud

There are [many ways](https://medium.com/@jaychapel/4-ways-to-get-google-cloud-credits-c4b7256ff862) to get promotional credits for Google Cloud. So I'm going to be using their services for this project. The plan is that everything that I will do on google cloud will be covered by those free credits :). 

The details of how to do this are included in google cloud's [documentation](https://cloud.google.com/run/docs/quickstarts/build-and-deploy). I'll be mainly following their docs. 

I created a [project](https://cloud.google.com/resource-manager/docs/creating-managing-projects) on google cloud and finally got the deployment to work. The google cloud docs are pretty well written - but I still ran into errors. I found [this repo](https://github.com/npatta01/web-deep-learning-classifier/) to be very helpful.

As usual I created a shell script to implement exactly what we need to do in terminal. 

```shell

# adapted from: https://github.com/npatta01/web-deep-learning-classifier/blob/master/docs/2_b_gcloud.md

GCP_PROJECT=fresh-runway-246001
APP_NAME=neelsmlapp
REGION="us-central1"
MEMORY=2G

# set project to correct one
gcloud config set project $GCP_PROJECT

# build 
gcloud builds submit --tag gcr.io/$GCP_PROJECT/$APP_NAME \
--timeout=82800

# run
gcloud beta run deploy $APP_NAME \
--image gcr.io/$GCP_PROJECT/$APP_NAME \
--region $REGION --memory $MEMORY --allow-unauthenticated

```

After implementing the render factor hack described above this web app finally works on google cloud. 






