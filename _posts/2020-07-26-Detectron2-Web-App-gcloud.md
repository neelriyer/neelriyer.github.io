---
layout: post
title:  Low Memory Instance Segmentation
---
 
Hacks to deploy an instance segmentation web app to google cloud

![alt text](/images/detectron2_web_app/basketball.jpeg)

Machine Learning models are memory intensive. My [current web app](https://spiyer99.github.io/Detectron2-Web-App/) consumes at least 1GB of memory. That makes it difficult to deploy to the cloud. 

I keep seeing the dreaded [out of memory error](https://en.wikipedia.org/wiki/Out_of_memory).

The immediate solution that comes to mind is increasing the memory of the VM instance. But I'd rather not spend more money that I have to. This is a side project, after all. 

So I came up with a work around. It's actually inspired by [deoldify](https://github.com/jantic/DeOldify/tree/master/deoldify) by Jason Antic. Jason uses a render factor to downscale the image and convert it to a square before running his machine learning model. 

I thought that applying the same idea here could reduce memory requirements. 

# The Solution

In short here is the solution. It's largely taken from [here](https://github.com/jantic/DeOldify/blob/master/deoldify/filters.py).

<script src="https://gist.github.com/spiyer99/fe661783ff3c8c222bd085ef03a2bb5e.js"></script>

We scale to square of size `targ`. `targ` is essentially the render factor that Jason mentions in deoldify. 

If the render factor is too high it can [lead to OOM errors](https://github.com/jantic/DeOldify/blob/edac73edf1d3557f95a71f860cffd6c4c91f66f0/deoldify/filters.py#L58). Too low and the result will have a poor resolution. 

What this looks like in practice is pretty simple actually. First we scale the image to a sqaure of size `targ`. Then we run inference on that scaled image. The result from inference is passed through the `_unsquare` function. This converts our square size image to its former size. 

<script src="https://gist.github.com/spiyer99/98511babdd831711f08174002ac03099.js"></script>

After this we can finally deploy on most cloud platforms without any OOM issues. If you do get an OOM error, just reduce the render factor. 

# Modifying the Web App

I'll be using the web app described [here](https://spiyer99.github.io/Detectron2-Web-App/) as a starting point.

We'll need to modify the `app.py` script to implement the `run_inference_transform` function described above. The modified version is on [Github](https://github.com/spiyer99/detectron2_web_app/blob/master/app.py).


# Deploying on Google Cloud

There are [many ways](https://medium.com/@jaychapel/4-ways-to-get-google-cloud-credits-c4b7256ff862) to get promotional credits for Google Cloud. So I'm going to be using their services for this project. The plan is that everything that I will do will be covered by those free credits :). 


I created a [project](https://cloud.google.com/resource-manager/docs/creating-managing-projects) on google cloud and finally got the deployment to work. The google cloud [docs](https://cloud.google.com/run/docs/quickstarts/build-and-deploy) are pretty well written - but I still ran into errors. I found [this repo](https://github.com/npatta01/web-deep-learning-classifier/) to be very helpful.

As usual I created a shell script to implement exactly what we need to do in terminal. 

<script src="https://gist.github.com/spiyer99/23ca4d121e2a1e1f11ab356ceac6fc79.js"></script>

Once we deploy this on google cloud the detectron2 model finally works. The output has a lower resolution than the input but everything fits within the 2G memory constraints. 

After building with `docker stats` I can confirm that this method uses far less memory. On my local computer it uses 1.16 GB of memory at its peak. Whereas the previous method used over 2GB (approx. 2.23 GB) at its peak. 

You can access the [google cloud instance segmentation web app here](https://neelsmlapp-lfoa57ljxa-uc.a.run.app/).  







