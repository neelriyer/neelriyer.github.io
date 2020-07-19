---
layout: post
title:  Instance Segmentation Web App
---

Building a Web App for Instance Segmentation using in Docker, Flask and Detectron2

{% include youtubePlayer.html id="VVLkB-vYmCc?autoplay=1&mute=1" %}

<iframe id="ytplayer" type="text/html" width="640" height="360"
  src="https://www.youtube.com/embed/VVLkB-vYmCc?autoplay=1"
  frameborder="0"
  width="1120" 
  height="630"></iframe>


Detectron2 offers state of the art instance segmentation models. It's very [quick to train](https://detectron2.readthedocs.io/notes/benchmarks.html) and offers very good results. 

Model training is fairly straightforward. There are many [tutorials](https://github.com/facebookresearch/detectron2/blob/master/GETTING_STARTED.md) to help you there. Deploying the model to a web app is a different story. When I was trying to do this I didn't find a lot of help on the internet. 

So, in this post we'll create a web app for detectron2's instance segmentation. 

# Backend

First, we'll create the machine learning backend. This will use basic [flask](https://flask.palletsprojects.com/en/1.1.x/). We'll start from some fairly standard [boilerplate code](https://github.com/realpython/flask-boilerplate/blob/master/app.py)

<script src="https://gist.github.com/spiyer99/bbf1ddea54ed9acc45e6aec398e97e54.js"></script>

This app will simply render the template `index.html`. I've specified the port manually. 

Next we'll add functions to get the image. We want to be able to upload an image to the website. We also want to be able to supply the website with a url and the image will be downloaded automatically. I've created the code do to exactly that below.

<script src="https://gist.github.com/spiyer99/ca7ef12a11406fa4b8d70a4ddbd9cade.js"></script>

This piece of code allows us to upload an image into the backend (POST request). Or we can supply the backend with a url and it will download the image automatically (GET request). The code also converts the image to a `jpg`. I couldn't do inference on a `png` image using detectron2. So we'll have to convert to a `jpg`.

If the code can't download the image for whatever reason - It will return the `failure.html` template. This will basically just be a simple `html` page saying there was an error in retrieving the image. 

Also, I've specified a different `@app.route` (/detect). This will need to refelcted in the `index.html` file. 


# Frontend

Now I'll create the frontend `html` code. Through this inferface the user can upload an image, and also specify a url to the image. 

<script src="https://gist.github.com/spiyer99/ba33eb7e2770473d74b358066cb0058e.js"></script>

There's not much to it. We create a simple form and tell it to link to the app.route('/detect') flask code. We also need to specify the method. If the user is uploading an image, it's POST. If the user is giving us the url to an image, it's GET. 

The `failure.html` template is even simpler. 

<script src="https://gist.github.com/spiyer99/c7e43b50659916b48be0c4e33e7af62f.js"></script>

Now we can move on the actual deep learning part. 


# The Model

In this part, we'll get a detectron2 pretrained model to do inference on an image. Then we'll link it to our existing backend. 

This part is slightly more involved. We'll create a new class called `Detector`. In that we'll create the `cfg` that is required for detectron2. Then we'll create another function that will do the inference on an image. 

I'll be using the mask rcnn pretrained model trained on the [Imagenet](http://www.image-net.org/) dataset. It will use a [ResNet](https://towardsdatascience.com/an-overview-of-resnet-and-its-variants-5281e2f56035)+[FPN](https://towardsdatascience.com/review-fpn-feature-pyramid-network-object-detection-262fc7482610) backbone. This model is said to obtain the [best speed/accuracy tradeoff](https://github.com/facebookresearch/detectron2/blob/master/MODEL_ZOO.md#common-settings-for-coco-models). 


<script src="https://gist.github.com/spiyer99/b6fe509cc94c959f9a66092aee4ae176.js"></script>


This piece of code pretty much does everything we need it to do for inference. We just need to specify the path to the pretrained model that we downloaded. 

It gets the config that corresponds our pretrained model automatically from the model zoo. It should also get the pretrained model from the model zoo as well. But I found that this doesn't really work in docker - at least for me. 

The next step is to integrate this `Detector` class into our existing script.

<script src="https://gist.github.com/spiyer99/97884765faad4dee8736dc005d8ae320.js"></script>


In this piece of code I've added in the `Detector` class we created previously. I've also created a function called `run_inference`. This is where the backend will run the detectron2 model on an image. It will take in an image path and call the detectron2 model through the `Detector` class we created earlier.

The `run_inference` function will return an image once instance segmentation is completed. I had to do some hacky stuff with the result fo the `run_inference` function to get it to work. `result_img` is pasted into a file object and that file object is returned as a png. There might be a better way to do this though. 


# Docker

The final step is creating a docker container for our code. Then we'll deploy the docker container locally. 

Thankfully, detectron2 has already created a [dockerfile](https://github.com/facebookresearch/detectron2/blob/master/docker/Dockerfile) for us. So we can work from that code. 

<script src="https://gist.github.com/spiyer99/4c014a23a903d9a57ec2baace40659b6.js"></script>

I've bascially used the dockerfile supplied in detectron2's [github repo](https://github.com/facebookresearch/detectron2/blob/master/docker/Dockerfile). But I made a few changes. 

I've added a [`requirements.txt`](https://github.com/spiyer99/detectron2_web_app/blob/master/requirements.txt) file. I do a `pip install` from that requirements file. That installs a few libaries that we need for this to work. I've also changed the command to start the `app.py` script we created earlier. That will start the flask application and render the `index.html` template. 

Now we can start the docker container. We can do this using the following:

```shell
docker build . -f Dockerfile -t detectron2 &&\
docker run -d -p 8080:8080 detectron2
```

That will build the current dockerfile in the current working directory. Then it will run that image on port 8080. That's the port that I specified earlier. 

But the issue with that if you keep running those commands over time you'll have too many docker images and containers on your computer. This can take up a [lot of space](https://jimhoskins.com/2013/07/27/remove-untagged-docker-images.html) on your computer.

[Jim Hoskins](https://jimhoskins.com/2013/07/27/remove-untagged-docker-images.html) has a pretty elegant solution to this, which I adapted for my purposes. 

I created a nice shell script which:
 - stops all the containers
 - removes untagged containers
 - builds a new container
 - runs the container on port 8080
 - shows the trailing logs for the container

This script is *incredibly* useful. 

<script src="https://gist.github.com/spiyer99/927fc75cf90bd0060f8073a71e4fc878.js"></script>

Run this script from terminal. It will build and run the detectron2 web app. The web app should show up at `localhost:8080` on your browser.

![alt text](/images/detectron2_web_app/detectron2_web_app.png)
![alt text](/images/detectron2_web_app/detect.jpeg)

It works great!

The code for this can be found on [github](https://github.com/spiyer99/detectron2_web_app). 



