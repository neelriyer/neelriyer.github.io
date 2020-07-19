---
layout: post
title:  Low Memory Instance Segmentation
---
 
Hacks to deploy an instance segmentation web app to google cloud


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
from ObjectDetector import Detector
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

After this we can finally deploy on most cloud platforms without any OOM issues. If you do get an OOM error, just reduce the render factor. 

# Modifying the Web App

I'll be using the web app described [here](https://spiyer99.github.io/Detectron2-Web-App/) as a starting point.

We'll need to modify the `app.py` script to implement the `run_inference_transform` function described above. 


```python
from ObjectDetector import Detector
import io
from flask import Flask, render_template, request, send_from_directory, send_file
from PIL import Image
import requests
import os
import img_transforms

app = Flask(__name__)
detector = Detector()

RENDER_FACTOR = 25

# function to load img from url
def load_image_url(url):
	response = requests.get(url)
	img = Image.open(io.BytesIO(response.content))
	return img

# run inference using image transform to reduce memory
def run_inference_transform(img_path = 'file.jpg', transformed_path = 'file_transformed.jpg'):

	# get height, width of image
	original_img = Image.open(img_path)

	# transform to square, using render factor
	transformed_img = img_transforms._scale_to_square(original_img, targ=RENDER_FACTOR*16)
	transformed_img.save(transformed_path)

	# run inference using detectron2
	untransformed_result = detector.inference(transformed_path)

	# unsquare
	result_img = img_transforms._unsquare(untransformed_result, original_img)

	# clean up
	try:
		os.remove(img_path)
		os.remove(transformed_path)
	except:
		pass

	return result_img

# run inference using detectron2
def run_inference(img_path = 'file.jpg'):

	# run inference using detectron2
	result_img = detector.inference(img_path)

	# clean up
	try:
		os.remove(img_path)
	except:
		pass

	return result_img


@app.route("/")
def index():
	return render_template('index.html')


@app.route("/detect", methods=['POST', 'GET'])
def upload():
	if request.method == 'POST':

		try:

			# open image
			file = Image.open(request.files['file'].stream)

			# remove alpha channel
			rgb_im = file.convert('RGB')
			rgb_im.save('file.jpg')
		
		# failure
		except:

			return render_template("failure.html")

	elif request.method == 'GET':

		# get url
		url = request.args.get("url")

		# save
		try:
			# save image as jpg
			# urllib.request.urlretrieve(url, 'file.jpg')
			rgb_im = load_image_url(url)
			rgb_im = rgb_im.convert('RGB')
			rgb_im.save('file.jpg')

		# failure
		except:
			return render_template("failure.html")


	# run inference
	result_img = run_inference_transform()
	# result_img = run_inference('file.jpg')

	# create file-object in memory
	file_object = io.BytesIO()

	# write PNG in file-object
	result_img.save(file_object, 'PNG')

	# move to beginning of file so `send_file()` it will read from start    
	file_object.seek(0)

	return send_file(file_object, mimetype='image/jpeg')


if __name__ == "__main__":

	# get port. Default to 8080
	port = int(os.environ.get('PORT', 8080))

	# run app
	app.run(host='0.0.0.0', port=port)
```

This builds from the previous [instance segmentation web app](https://spiyer99.github.io/Detectron2-Web-App/) I created. Now it adds a layer where it 

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

Once we deploy this on google cloud the detectron2 model finally works. The output has a lower resolution than the input but everything fits within the 2G memory constraints. 

After building with `docker stats` I can confirm that this method uses far less memory. On my local computer it uses 1.16 GB of memory at its peak. Whereas the previous method used over 2GB (approx. 2.23 GB) at its peak. 

You can access the final instance segmentation web app here: 










