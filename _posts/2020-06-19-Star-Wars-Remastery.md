---
layout: post
title: Remastering Star Wars using Deep Learning
---
<!-- 

I’m a huge Star Wars fan. And like a lot of Star Wars fans I’ve been getting into Star Wars clone wars episodes on cartoon network and disney. I might be too old for it, but it’s a phenomenal show. 

But when I go back to watch the old movies I’m annoyed by the drop in quality. Why does Obi Wan’s skin look papery? Why do the space scenes look terrible? Why does the armour on storm troopers look so ‘boxy’?

Don’t get me wrong I’m a huge fan of Star Wars. And for the most part I was more than happy to overlook these small imperfections when re-watching ‘A New Hope’. 

But that all changed when I watched the deleted scenes. 

(Show clip here)

What the hell are these black specs that keep appearing on my screen when I’m watching a movie? Why are they all different shapes and sizes? What are they?

I googled them. Apparently they’re [cue marks](https://en.wikipedia.org/wiki/Cue_mark). Marks that come from scratches on film. Film! The original Star Wars was created on film. I forget just how old this franchise is sometimes.

So having way to much time on my hands (and not having a girlfriend) I decided why can't I build something to fix this. 

Video restoration is a difficult field to get into. A bit of googling had me all over the place. Photo restoration is done manually for the most part. People use photoshop and a variety of different image editing tools and restore old photos. You can find them [here](https://www.reddit.com/r/estoration/). 

But video restoration is another matter altogether. We would need to restore each frame in the video on photoshop or any other image editing tool.

Big studios do some magic with the original film used to create the movie. But I don’t have the original film. I just have the video from youtube.

But could we apply deep learning? Probably. It’s worth a shot. 

So here’s what I did. I downloaded high quality videos into from youtube. Then I ruined them. I added black specs and reduced the resolution of the video. [Ffmpeg](https://ffmpeg.org/) was very useful in doing this. 

Now I had two videos. One in perfect quality and another in shitty quality.

Then I extracted frames from each video. Initially I adopted a naive approach for doing this. Where I would do through the video in python and scrape each frame individually. But that took too long. I used multi-processing here to really speed things up.
Great. Now I had two datasets. One of shitty quality images (taken from the ruined video) and one of good quality images (taken from the high quality video). 

Then I trained the [NoGAN network](https://www.fast.ai/2019/05/03/decrappify/) pioneered by fastai and jason antic on this data.

I trained the model on google colab’s free gpus. They’re a great resource and I can’t believe they are free. The interesting thing that fastai recommends is increasing the size of your images gradually. So at first you train on small size images, then you upscale your images and retrain on the larger images. It saves you a lot of time. Pretty smart. 

Once this had trained, I ran inference on the model. This was more involved than I originally thought. I had to download the Star Wars deleted scenes (using [youtube-dl](https://github.com/ytdl-org/youtube-dl)) and then extract all the frames in this video. 

(insert frame here)

Then I had to run inference from the learner on each individual frame of the video. That takes a long time. 

(insert inference image here)

Then I had to stitch all these frames together to create a video. Here is the final output.

(show video here)

As you can see there is room for improvement. The sky needs a bit more work. But I like the vibrancy of the background. That is an interesting (and completely unplanned) effect. The goal was to remove the ‘cue marks’ (annoying black specs) from the video. I think its done okay in that respect.

In the future, I’d like to improve the sky. I’m also thinking of doing more super-resolution on the video. It would be nice to show a young Luke Skywalker in high quality. Training is still a lengthy process for me. I need to work on reducing this.

Inference is also an issue. It tends to overload memory and crash. The result is that ___ is the longest I could get for this video. Not completely sure how to solve this problem. But I'll need to solve it if I'm going to be using this further.



 -->