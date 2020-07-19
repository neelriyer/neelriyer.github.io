---
layout: post
title:  Detectron2 Web App 
---

__Building a Web App in Docker and Flask for Detectron2__

(show video here)

Detectron2 offers state of the art instance segmentation models. It's very [quick to train](https://detectron2.readthedocs.io/notes/benchmarks.html) and offers very good results. 

Model training is is fairly straightforward. There are many [tutorials](https://github.com/facebookresearch/detectron2/blob/master/GETTING_STARTED.md) to help you there. Deploying the model to a web app is a different story. In this post we'll create a web app for detectron2's instance segmentation. 

# Backend

First, we'll create the machine learning backend. This will use basic [flask](https://flask.palletsprojects.com/en/1.1.x/). We'll start from some fairly standard [boilerplate code](https://github.com/realpython/flask-boilerplate/blob/master/app.py)

```python
import io
from flask import Flask, render_template, request, send_from_directory, send_file
from PIL import Image
import requests
import os
import urllib.request

app = Flask(__name__)

@app.route("/")
def index():

	# render the index.html template
	return render_template('index.html')

if __name__ == "__main__":

	# get port. Default to 8080
	port = int(os.environ.get('PORT', 8080))

	# run app
	app.run(host='0.0.0.0', port=port)

```
This app will simply render the template `index.html`. I've specified the port manually. 

Next we'll add functions to get the image. We want to be able to upload an image to the website. We also want to be able to supply the website with a url and the image will be downloaded automatically. I've created the code do to exactly that below.

```python
import io
from flask import Flask, render_template, request, send_from_directory, send_file
from PIL import Image
import requests
import os

# function to load img from url
def load_image_url(url):
	response = requests.get(url)
	img = Image.open(io.BytesIO(response.content))
	return img

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

	return send_file(rgb_im, mimetype='image/jpeg')

```
This piece of code allows us to upload an image into the backend (POST request). Or we can supply the backend with a url and it will download the image automatically (GET request). The code also converts the image to a `jpg`. I couldn't do inference on a `png` image using detectron2. So we'll have to convert to a `jpg`.

If the code can't download the image for whatever reason - It will return the `failure.html` template. This will basically just be a simple `html` page saying there was an error in retrieving the image. 

Also, I've specified a different `@app.route` (/detect). This will need to refelcted in the `index.html` file. 

# Frontend

Now I'll create the frontend `html` code. Through this inferface the user can upload an image, and also specify a url to the image. 

```
<!DOCTYPE html>
<html lang="en">

<body>

<h1 style="text-align:center;">Detectron2 Web App</h1>
<br>
<h2>Detectron2 Instance Segmentation</h2>

<form action = "/detect" method = "POST" enctype = "multipart/form-data">
	<input type = "file" name = "file" />
	<input name = "submit" type = "submit"/>
</form>
<form action = "/detect" method = "GET" enctype = "multipart/form-data">
	<input type="text" name="url">
	<input type = "submit"/>
</form>
```

There's not much to it. We create a simple form and tell it to link to the `app.route('/detect')` flask code. We also need to specify the method. If the user is uploading an image, it's POST. If the user is giving us the url to an image, it's GET. 

The `failure.html` template is even simpler. 

```
{% block content %}
<body>

    <p> Error in retrieving image </p>

</body>
{% endblock %}
```

Now we can move on the actual deep learning part. 

# The Model

In this part, we'll get a detectron2 pretrained model to do inference on an image. Then we'll link it to our existing backend. 

This part is slightly more involved. We'll create a new class called `Detector`. In that we'll create the `cfg` that is required for detectron2. Then we'll create another function that will do the inference on an image. 

I'll be using the mask rcnn pretrained model trained on the [Imagenet](http://www.image-net.org/) dataset. It will use a [ResNet](https://towardsdatascience.com/an-overview-of-resnet-and-its-variants-5281e2f56035)+[FPN](https://towardsdatascience.com/review-fpn-feature-pyramid-network-object-detection-262fc7482610) backbone. This model is said to obtain the [best speed/accuracy tradeoff](https://github.com/facebookresearch/detectron2/blob/master/MODEL_ZOO.md#common-settings-for-coco-models). 

This model is trained on a 3x schedule, ~37 COCO epochs.
You'll have to [download the model](https://dl.fbaipublicfiles.com/detectron2/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl)



