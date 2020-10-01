---
layout: post
title: CycleGAN Math in Simple English
---

{% include youtubePlayer.html id="9reHvktowLY?autoplay=1&mute=1&loop=1&playlist=9reHvktowLY" %}

<!-- for mathjax support -->
<script type="text/x-mathjax-config">
MathJax.Hub.Config({
TeX: { equationNumbers: { autoNumber: "AMS" } }
});
</script>
<script type="text/javascript" async src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML"></script>


![examples](https://junyanz.github.io/CycleGAN/images/objects.jpg)

CycleGAN is a method of unpaired image to image translation. Unfortunately, its possible to use CycleGAN without fully understanding or appreciating the mathematics involved. That is a real shame. 

In this article I'll walkthrough the mathematics behind Cycle-Consistent Adversarial Networks. Please read the [paper](https://arxiv.org/pdf/1703.10593.pdf) for a more comprehensive explanation.


# Unpaired vs Paired

The key thing with CycleGAN is that we don't have before and after images. 

Let's take the example shown above of converting a zebra into a horse (and vice-versa).

In a paired dataset the horse and zebra need to "match" each other. We're essentially taking a horse and painting it black and white. The background, lightning, etc. stays the same.

A paired dataset would look something like this:

![examples](https://miro.medium.com/max/4604/1*5DG4hHjxAyWTfV1J3mRH_A.png)

In an unpaired dataset the horses and zebras don't need to "match" each other. This is dataset is far easier to create. 

And an unpaired dataset would look something like this like this:

![examples](http://www.dallasequestriancenter.com/wp-content/uploads/hvsz.jpg)


# Generators and Discriminators

As you probably expect for a GAN we have generators and discriminators. Our generator is also known as the mapping function. 

Let's name a few variables:

*Horses and Zebras*

- $X$ refers to a Zebra
- $Y$ refers to a Horse

*Generators*

- $G$ refers to a mapping function that converts a Zebra into a Horse
- $F$ refers to a mapping function that converts a Horse into a Zebra

*Discriminators*

- $D_{x}$ refers to a discriminator that is really good at recognising Zebras.
- $D_{y}$ refers to a discriminator that is really good at recognising horses

Putting that all together we have something that looks like this:

<center>
<img src="https://drive.google.com/uc?id=1GjhJKmBdZNC6lGYkREGratSQ-36IAKwk" align="center" width="200" />
</center>


# Cycle Consistency Loss 

In the [paper](https://arxiv.org/pdf/1703.10593.pdf) the diagram for Cycle Consistency is as follows:

<center>
<img src="https://drive.google.com/uc?id=1qaYsaQrVchH5NjkGKOQ7A6a31nG75Syx" align="center" width="400" />
</center>

Here's what's happening: We take an image of a horse and convert it into a zebra. Then run the discriminator on this newly created zebra. Then we take our newly created zebra and convert it into a horse. 

We compare this newly created horse to our existing horse. This is what $\hat{x}$ vs $x$ essentially means. Our newly created horse should look almost identical to original horse. 

I've created an infographic to explain how this works.

<img src="/images/cyclegan_simple/infographic.png" alt="img"/> 

Now we repeat this process. Except this time we taken an image of a zebra and try to convert that into a horse. 


# Loss Functions in Mathematical Terms

## Adversial loss: 

The loss function for adversial loss is as follows:

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

The RHS features this term:

<center>
<img src="https://drive.google.com/uc?id=1iRior0WTK5eOamzMIuVw-d6472bSbBAz" width="200"/>
</center>

This term measures our ability to recognise whether our created zebra is real or fake. We draw one zebra out of all the zebras and pass it through the discriminator.

Now for the next term:

<center>
<img src="https://drive.google.com/uc?id=1WM6QtgZf6WRqFYj_lSSXXkyNRjIxO2QP" width="250"/>
</center>

This term measures our ability to take a horse and turn it into a zebra. We draw one horse out of all the generated horses and pass it through the discriminator.

Now we do the first term plus the second term. This gives us the adversial loss equation we saw above.


## Cycle Consistency Loss: 

Here's the loss function for cycle consistency loss:

<center>
<img src="https://drive.google.com/uc?id=1Hr12uqBUfjwskg0TZ0LuPZELIG_X1AOo" width="400"/>
</center>


Let's look at the first term:

<center>
<img src="https://drive.google.com/uc?id=1rfj_Z_mPYc334nXqpJqJc7QIcwPO64Rb" width="250"/>
</center>

This means:

1. We take a random zebra from our zebra dataset. (x~p(x))
2. We pass that zebra through the generator ($G$) to create a horse. 
3. We pass that generated horse through another generator ($F$) to create a zebra
4. We compare the zebra created in step 3 with the random zebra in step 1 and take the sum of the absolute value of differences. 


The second term works in a very similar fashion. However, now we start with a random horse and convert that into a zebra and back into a horse again.


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

The mathematics behind CycleGAN can seem somewhat daunting. But I've tried to cover the broad strokes in this article. I hope this has helped someone out there. This article would've certaintly helped me when I was learning how CycleGAN worked.

If I've made a mistake please feel free to reach out to me on twitter and I'll fix it as soon as I can.

A big thanks to the authors of CycleGAN: [Jun-Yan Zhu](https://www.cs.cmu.edu/~junyanz/), [Taesung Park](https://taesung.me/), [Phillip Isola](http://web.mit.edu/phillipi/) and [Alexei A. Efros](https://people.eecs.berkeley.edu/~efros/). I think it's worth reading their [paper](https://arxiv.org/pdf/1703.10593.pdf) to understand this method in more detail. 








