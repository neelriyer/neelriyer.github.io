---
layout: post
title:  Instance Segmentation Web App- Part 2
---

Reducing the Memory required for inference

Machine Learning Models are memory intensive. Even for running inference my [current web app](https://spiyer99.github.io/Detectron2-Web-App/) at least 1 GB of memory. That makes it difficult to deploy to the cloud. 

When deploying to google cloud I kept running into this issue. [Out of memory error]. 

The immediate solution that comes to mind is increasing the memory of the VM instance. But I'd rather not spend more money that I have to. This is a side project, after all. 

So I came up with a work around. It's actually inspired by [deoldify](https://github.com/jantic/DeOldify/tree/master/deoldify) by Jason Antic. Jason uses a render factor to downscale the image and convert it to a square before running his machine learning model. 

I thought that applying the same idea here could reduce the memory requirements. 

# The Hack









It turns out detectron2 just takes up too much room for heroku. That leaves us with AWS and google cloud. Cause I'm pretty poor - I'm going to be using the option that is free or very close to free. 

There are [various hacks](https://medium.com/@jaychapel/4-ways-to-get-google-cloud-credits-c4b7256ff862) to get promotional credits for Google Cloud. So I'm going to be using their services for this tutorial. The plan is that everything that I will do on google cloud will be covered by those free credits :). 

The details of how to do this are included in google cloud's [documentation](https://cloud.google.com/run/docs/quickstarts/build-and-deploy). I'll be mainly following their docs. 


# Deployment

I created a [project](https://cloud.google.com/resource-manager/docs/creating-managing-projects) on google cloud and finally got the deployment to work. The google cloud docs are pretty well written - but I still ran into errors. I found [this repo](https://github.com/npatta01/web-deep-learning-classifier/) to be very helpful. It basically implements exactly what we need to do on google cloud.

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

After deploying this to google cloud we run into an issue. Detectron2 uses far too much memory for the 2G that we've allocated. A quick google search [reveals](https://cloud.google.com/run/docs/configuring/memory-limits) that 2G is the maximum that can be used by a Cloud run.

We could upgrade to a VM instance with more memory. But like I said I'd rather not spend money unless I really have to. So I've come up with a solution. 

# The Hack






