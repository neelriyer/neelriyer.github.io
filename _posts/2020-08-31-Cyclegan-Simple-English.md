---
layout: post
title: Cyclegan in Simple Terms
---

{% include youtubePlayer.html id="9reHvktowLY?autoplay=1&mute=1&loop=1&playlist=9reHvktowLY" %}

{% include mathjax.html %}

![examples](https://junyanz.github.io/CycleGAN/images/objects.jpg)

Cyclegan is the method that is responsible for the creations shown above. It's a method of unpaired image to image translation. In this article I'll describe how Cyclegan works in simple(ish) terms.


# Unpaired vs Paired

Cyclegan aims to perform unpaired image translation. The key thing is that we don't have before and after images. 

Let's take the example shown above of converting a zebra into a horse (and vice-versa).

If we needed paired images we would have to take a photo of a horse, then paint it black and white, and take a photo of our "zebra". Both photos would need to be identical- except in one we would have a zebra and in the other we would have a horse. The background, the lighting, etc. would all need to be the same. 


An unpaired dataset is much easier to create. In this dataset we just need photos of horses and photos of zebras. Our horses and zebras don't need to "match" each other.


## Generators and Discriminators

As you probably expect for a GAN we have generators and discriminators. Our generator is also known as the mapping function.

<center>
<img src="https://drive.google.com/uc?id=1GjhJKmBdZNC6lGYkREGratSQ-36IAKwk" align="center" width="200" />
</center>

Suppose a Zebra is denoted as $X$ and a Horse is denoted as $Y$.

We have a mapping function which converts a Zebra to a Horse (call that $G$). Another mapping function that converts Horse to a Zebra (call that $F$). 

We have a discriminator that is really good at recognising zebras. We call that $D_{x}$. We have another discriminator that is really good at recognising horses. We call that $D_{y}$.


## Cycle Consistency Loss 

<center>
<img src="https://drive.google.com/uc?id=1qaYsaQrVchH5NjkGKOQ7A6a31nG75Syx" align="center" width="400" />
</center>


Here's what this loss function is doing: We taken an image of a horse and convert it into a zebra. Then run the discriminator on this newly created zebra. Then we take our newly created zebra and convert it into a horse. 

We compare this newly created horse to our existing horse. This is what $\hat{x}$ vs $x$ essentially means. Our newly created horse should look almost identical to original horse. 

<center>
<img src="https://drive.google.com/uc?id=18rpwM3DWBUdUv9EIKGeTvRaZw1F8Puwa" align="center" width="400" />
</center>

Now we do almost exactly the same thing again. Except this time we taken an image of a zebra and try to convert it to a horse. 


## Loss functions

For our loss functions we have adversial loss and cycle consistency loss.

### Adversial loss: 

<center>
<img src="https://drive.google.com/uc?id=1c5dVq2K_9OFuv77oKPhcq0jtTGW2_Guu" align="center" width="400" />
</center>

<!-- 
y~pdata(y) -> means grab a random horse from our horse dataset. -->

Let's break this down. 

The left hand side takes in:
- $D_{y}$ (horse recogniser)
- $G $(our zebra to horse creator)
- $X$ (a zebra) 
- $Y$ (a horse)

The right hand side features this term:

<center>
<img src="https://drive.google.com/uc?id=1iRior0WTK5eOamzMIuVw-d6472bSbBAz" width="200"/>
</center>

This term measures our ability to draw one zebra out of all the zebras. This equation tries to recognise whether our created zebra is real or fake.

The right hand side also features this term:

<center>
<img src="https://drive.google.com/uc?id=1WM6QtgZf6WRqFYj_lSSXXkyNRjIxO2QP" width="250"/>
</center>

This particular term measures our ability to take a horse and turn it into a zebra. This equation tries to recognise whether our created zebra is real or fake.

Now we do the first term minus the second term. Ignoring the logs this gives us the adversial loss equation we saw above.


### Cycle Consistency Loss: 


<center>
<img src="https://drive.google.com/uc?id=1Hr12uqBUfjwskg0TZ0LuPZELIG_X1AOo" width="400"/>
</center>


If we look at the first term:

<center>
<img src="https://drive.google.com/uc?id=1rfj_Z_mPYc334nXqpJqJc7QIcwPO64Rb" width="250"/>
</center>

This means:

1. We take a random zebra from our zebra dataset. 
2. We pass that zebra through the generator ($G$) to create a horse. 
3. We pass that generated horse through another generator ($F$) to create a zebra
4. We compare the zebra created in step 3 with the random zebra in step 1 and take the sum of the absolute value of differences. 

The second term works in a very similar fashion. Except now we take out a random horse and pass that through the generator functions.


## Full objective

The full loss function is as follows:

<center>
<img src="https://drive.google.com/uc?id=10aViByAIMguwjyrWQBjZeAr16egUlVXI" align="center" width="400" />
</center>

It's just the sum of the Adversial loss functions we saw earlier and the cycle consistency loss functions. Where $λ$ controls the relative importance of each objective.

Ultimately we want to achieve the following:


<center>
<img src="https://drive.google.com/uc?id=1cfEWGvboitsqmwDyC0fSEjE0PDvT1hvl" align="center" width="400" />
</center>


We try to maximise the capability of the discriminator and minimise the loss of the generator. This allows us to get very good at differentiating between zebras and horses, while also getting very good at generating both horses and zebras.


# End

I've covered the broad strokes of how Cyclegan works. If I've made a mistake here, or if anything seems incorrect please let me know and I'll fix it as soon as I can.

A big thanks to the creators of Cyclegan. I think it's worth reading their [paper](https://arxiv.org/pdf/1703.10593.pdf) to understand this method in more detail. 








