---
layout: post
title: Pool Detection from Aerial Imagery
---
![alt text](/images/aerial_object_detection/meme.jpg)

<!-- <sub>Image by Nick Dambrosia on [Unsplash](https://unsplash.com/photos/XN1wsJMh2lo)</sub> -->

__Unfinished post__

There's a lot of talk about swimming pool detection from aerial imagery.

You're probably interested in a code first example. I was too. But I couldn't find one.

I decided to make my own.

It's not perfect. It's not pretty. But it seems to work.

All code is on [Github](https://github.com/spiyer99/aerial_object_detection). Criticism is appreciated. 

# Dataset

To make this you'll need data. Lots of labelled training data. This can be tricky to obtain. Particularly when your budget is as low as mine ($0). 

I managed to find a [government resource](https://maps.six.nsw.gov.au/) that gives you high quality aerial imagery. The resolution is significantly better than google maps or google earth. If you're looking for free high resolution aerial imagery this is probably your best bet. 

I also managed to find a dataset on [kaggle](https://www.kaggle.com/kbhartiya83/swimming-pool-and-car-detection). This was useful too.

# Model

Initially I created the model using [icevision](https://github.com/airctic/icevision/). It's this new deep learning framework that integrates well with fastai and pytorch. But the results were less than stellar. Maybe I'm doing something wrong. Maybe icevision is just not that good. Who knows? 

I'm going to keep tinkering with this though. I think icevision has potential.

I also experimented with Detectron2 from facebook research. This performed much better straight out of the box. However I wasn't able to easily implement things like test time augmentation, Mixup, learning rate finder, etc. Fastai (and subsequently icevision) makes these things incredibly simple.

I'll be using the detectron2 model for now.

# Results

Here's some successes.

![alt text](/images/aerial_object_detection/success.png)

And here's some failures. 

![alt text](/images/aerial_object_detection/failure.png)

# Deployment

I deployed the model using docker. I describe how I make this in a [previous article](https://spiyer99.github.io/Detectron2-Web-App/).

If I can improve the model I might deploy this to google cloud so everyone can use it.

Again the code for deployment is on [Github](https://github.com/spiyer99/aerial_object_detection).


# Conclusion

This is really just a first step. Please feel free to contribute and make this better. 

I would really like to implement the relative luminance transform in icevision. [Apparently](https://towardsdatascience.com/weekend-project-detecting-solar-panels-from-satellite-imagery-f6f5d5e0da40) it is very useful for this kind of thing. But icevision currently [converts](https://github.com/airctic/icevision/blob/5a92bcd0ec8aa791ce9f37aded7763b09fe0e8be/icevision/utils/imageio.py#L13) the image to a RGB. This makes it tricky to implement a transform on the alpha channel.

If you liked this or found it useful, please let me know on [twitter](https://twitter.com/neeliyer11).



