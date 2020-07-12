---
layout: post
title: Remastering Star Wars using Deep Learning
---
A New Hope for the Deleted Scenes.

I’m a huge Star Wars fan. And like a lot of Star Wars fans I’ve been getting into [Star Wars: The Clone Wars](https://www.imdb.com/title/tt0458290/) on Cartoon Network and Disney+. It’s a phenomenal show. 

But I'm always annoyed by the drop in video quality when I watch the older stuff. For example, here are the deleted scenes from  Star Wars: Episode IV: A New Hope (1977). This was the very first Star Wars to be created.

{% include youtubePlayer.html id="f00IkrWvur4?start=159" %}

What are those weird black specs that keep popping up? They really ruin the experience. Small wonder why these are the *deleted* scenes. 

Apparently those weird specs are called [cue marks](https://en.wikipedia.org/wiki/Cue_mark). They're marks that come from scratches on film. Star Wars is a fantastic series, but it’s also fantastically *old*. 

Deep Learning has recently been used for video restoration. The results have been very promising. [Deoldify](https://github.com/jantic/DeOldify) for example, allows users to colorize old videos and images. [NVIDIA's Noise2Noise model](https://www.youtube.com/watch?v=P0fMwA3X5KI) allows people to restore old images to their former glory. 

But so far there's nothing I know of that can specifically remove 'cue marks' and grainy spots from old film. So let's build it!

# Creating the Dataset

Creating the dataset was tricky - but still doable. Here's what I did. I downloaded high quality videos into from youtube. Then I ruined them. I added black specs and reduced the resolution of the video. [Ffmpeg](https://ffmpeg.org/) was very useful in doing this. 

First we'll download the video.

```shell
youtube-dl --format best -o seinfeld.mp4 https://www.youtube.com/watch?v=nEAO60ON7yo 
```

I'm using this video. I'm using a clip from [seinfeld](https://en.wikipedia.org/wiki/Seinfeld). Cause why not?

{% include youtubePlayer.html id="nEAO60ON7yo " %}

Then we'll need to ruin it. To do this I downloaded a grainy film overlay from youtube. Then I overlayed the video using ffmpeg with the blend setting set to [``softlight``](https://ffmpeg.org/ffmpeg-filters.html#blend-1). Finding the right blend setting took a lot of trial and error. The ffmpeg [docs](https://ffmpeg.org/documentation.html) don't have a lot of examples. 

```shell
# download grain video
rm -Rf build
YT_GRAIN_OVERLAY="https://www.youtube.com/watch?v=J_MZb7qTenE"
mkdir -p build
youtube-dl "$YT_GRAIN_OVERLAY" -f mp4 --output "build/grain.mp4"

# invert colors
ffmpeg -loglevel quiet -y -i "build/grain.mp4" -vf negate 'color_inverted.mp4'

# overlay video
ffmpeg \
	-y \
	-loglevel quiet \
	-i "seinfeld.mp4" \
	-i "color_inverted.mp4" \
	-filter_complex "[1:v][0:v]blend=all_mode='softlight':all_opacity=1" \
	"output_test.mp4"
```

Now we have two videos. One in perfect quality and another in shitty quality.

{% include youtubePlayer.html id="l8Z3T9w0yBY " %}
{% include youtubePlayer.html id="MaIDJO5ar1c " %}


Now we'll extract frames from each video. Initially I adopted a naive approach for doing this. Where I would do through the video in python and scrape each frame individually. But that took too long. 

We can use multi-processing here to really speed things up. This was adapted from [Hayden Faulker's](https://gist.github.com/HaydenFaulkner/54318fd3e9b9bdb66c5440c44e4e08b8#file-video_to_frames-py) script.

```python
# from https://gist.github.com/HaydenFaulkner/54318fd3e9b9bdb66c5440c44e4e08b8#file-video_to_frames-py

from concurrent.futures import ProcessPoolExecutor, as_completed
import cv2
import multiprocessing
import os
import sys


def print_progress(iteration, total, prefix='', suffix='', decimals=3, bar_length=100):
	"""
	Call in a loop to create standard out progress bar

	:param iteration: current iteration
	:param total: total iterations
	:param prefix: prefix string
	:param suffix: suffix string
	:param decimals: positive number of decimals in percent complete
	:param bar_length: character length of bar
	:return: None
	"""

	format_str = "{0:." + str(decimals) + "f}"  # format the % done number string
	percents = format_str.format(100 * (iteration / float(total)))  # calculate the % done
	filled_length = int(round(bar_length * iteration / float(total)))  # calculate the filled bar length
	bar = '#' * filled_length + '-' * (bar_length - filled_length)  # generate the bar string
	sys.stdout.write('\r%s |%s| %s%s %s' % (prefix, bar, percents, '%', suffix)),  # write out the bar
	sys.stdout.flush()  # flush to stdout


def extract_frames(video_path, frames_dir, overwrite=False, start=-1, end=-1, every=1):
	"""
	Extract frames from a video using OpenCVs VideoCapture

	:param video_path: path of the video
	:param frames_dir: the directory to save the frames
	:param overwrite: to overwrite frames that already exist?
	:param start: start frame
	:param end: end frame
	:param every: frame spacing
	:return: count of images saved
	"""

	video_path = os.path.normpath(video_path)  # make the paths OS (Windows) compatible
	frames_dir = os.path.normpath(frames_dir)  # make the paths OS (Windows) compatible

	video_dir, video_filename = os.path.split(video_path)  # get the video path and filename from the path

	assert os.path.exists(video_path)  # assert the video file exists

	capture = cv2.VideoCapture(video_path)  # open the video using OpenCV

	if start < 0:  # if start isn't specified lets assume 0
		start = 0
	if end < 0:  # if end isn't specified assume the end of the video
		end = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

	capture.set(1, start)  # set the starting frame of the capture
	frame = start  # keep track of which frame we are up to, starting from start
	while_safety = 0  # a safety counter to ensure we don't enter an infinite while loop (hopefully we won't need it)
	saved_count = 0  # a count of how many frames we have saved

	while frame < end:  # lets loop through the frames until the end

		_, image = capture.read()  # read an image from the capture

		if while_safety > 500:  # break the while if our safety maxs out at 500
			break

		# sometimes OpenCV reads None's during a video, in which case we want to just skip
		if image is None:  # if we get a bad return flag or the image we read is None, lets not save
			while_safety += 1  # add 1 to our while safety, since we skip before incrementing our frame variable
			continue  # skip

		if frame % every == 0:  # if this is a frame we want to write out based on the 'every' argument
			while_safety = 0  # reset the safety count
			save_path = os.path.join(frames_dir, "{:010d}.jpg".format(frame))  # create the save path
			if not os.path.exists(save_path) or overwrite:  # if it doesn't exist or we want to overwrite anyways
				cv2.imwrite(save_path, image)  # save the extracted image
				saved_count += 1  # increment our counter by one

		frame += 1  # increment our frame count

	capture.release()  # after the while has finished close the capture

	return saved_count  # and return the count of the images we saved


def video_to_frames(video_path, frames_dir, overwrite=False, every=1, chunk_size=1000):
	"""
	Extracts the frames from a video using multiprocessing

	:param video_path: path to the video
	:param frames_dir: directory to save the frames
	:param overwrite: overwrite frames if they exist?
	:param every: extract every this many frames
	:param chunk_size: how many frames to split into chunks (one chunk per cpu core process)
	:return: path to the directory where the frames were saved, or None if fails
	"""

	video_path = os.path.normpath(video_path)  # make the paths OS (Windows) compatible
	frames_dir = os.path.normpath(frames_dir)  # make the paths OS (Windows) compatible

	video_dir, video_filename = os.path.split(video_path)  # get the video path and filename from the path

	# make directory to save frames, its a sub dir in the frames_dir with the video name
	# os.makedirs(os.path.join(frames_dir, video_filename), exist_ok=True)

	capture = cv2.VideoCapture(video_path)  # load the video
	total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))  # get its total frame count
	capture.release()  # release the capture straight away

	if total < 1:  # if video has no frames, might be and opencv error
		print("Video has no frames. Check your OpenCV + ffmpeg installation, can't read videos!!!\n"
			  "You may need to install OpenCV by source not pip")
		return None  # return None

	frame_chunks = [[i, i+chunk_size] for i in range(0, total, chunk_size)]  # split the frames into chunk lists
	frame_chunks[-1][-1] = min(frame_chunks[-1][-1], total-1)  # make sure last chunk has correct end frame

	prefix_str = "Extracting frames from {}".format(video_filename)  # a prefix string to be printed in progress bar

	# execute across multiple cpu cores to speed up processing, get the count automatically
	with ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:

		futures = [executor.submit(extract_frames, video_path, frames_dir, overwrite, f[0], f[1], every)
				   for f in frame_chunks]  # submit the processes: extract_frames(...)

		for i, f in enumerate(as_completed(futures)):  # as each process completes
			print_progress(i, len(frame_chunks)-1, prefix=prefix_str, suffix='Complete')  # print it's progress

	return os.path.join(frames_dir, video_filename)  # when done return the directory containing the frames

```


Great. Now we have two datasets. One of crappy quality images (taken from the ruined video) and one of good quality images (taken from the high quality video). To make the crappy images crappier, I'll downscale them (this isn't a necessary step though).

```python
def resize_one(img, size):

	targ_sz = resize_to(img, size, use_min = True)
	img = img.resize(targ_sz, resample = PIL.Image.BILINEAR).convert('RGB')
	return img
```

This is what the crappy and normal images looked like now. Side note: this is a great scene from seinfeld.

![alt text](/images/star_wars/crappy_vs_clean_comparison.png)
![alt text](/images/star_wars/crappy_vs_clean_comparison1.png)

A quick check shows that we have a dataset of about `10014` files. Pretty good.


# Neural Network

Let's make the most of those `10014` files by using transforms.

I added horizontal and vertical flips, zoom changes, lighting changes and rotation changes. With Fastai this is really easy to do.

```python
bs,size=64,128
# bs,size=8,480
arch = models.resnet34

src = ImageImageList.from_folder(path_lr).split_by_rand_pct(0.1, seed=42)

tfms = get_transforms(do_flip=True, flip_vert=True, max_zoom = 1.1, max_lighting=0.2, max_rotate = 10)

def get_data(bs,size=None):

	if(size is None):
	  data = (src.label_from_func(lambda x: path_hr/x.name)
			.transform(tfms, tfm_y=True)
			.databunch(bs=bs).normalize(imagenet_stats, do_y=True))
	else:
	  data = (src.label_from_func(lambda x: path_hr/x.name)
			.transform(tfms,size=size, tfm_y=True)
			.databunch(bs=bs).normalize(imagenet_stats, do_y=True))

	data.c = 3
	return data

data = get_data(bs,size)

```

Here are some of the image transforms.

![alt text](/images/star_wars/transforms.png)

Not bad! 

We'll use the [NoGAN network](https://www.fast.ai/2019/05/03/decrappify/) pioneered by fastai and jason antic on this data. This code was inspired by [lesson 7](https://github.com/fastai/course-v3/blob/master/nbs/dl1/lesson7-superres.ipynb) of the fastai course.

```python
# taken from: https://github.com/fastai/course-v3/blob/master/nbs/dl1/lesson7-superres.ipynb

t = data.valid_ds[0][1].data
t = torch.stack([t,t])

def gram_matrix(x):
	n,c,h,w = x.size()
	x = x.view(n, c, -1)
	return (x @ x.transpose(1,2))/(c*h*w)

base_loss = F.l1_loss

vgg_m = vgg16_bn(True).features.cuda().eval()
requires_grad(vgg_m, False)

blocks = [i-1 for i,o in enumerate(children(vgg_m)) if isinstance(o,nn.MaxPool2d)]
blocks, [vgg_m[i] for i in blocks]

class FeatureLoss(nn.Module):
	def __init__(self, m_feat, layer_ids, layer_wgts):
		super().__init__()
		self.m_feat = m_feat
		self.loss_features = [self.m_feat[i] for i in layer_ids]
		self.hooks = hook_outputs(self.loss_features, detach=False)
		self.wgts = layer_wgts
		self.metric_names = ['pixel',] + [f'feat_{i}' for i in range(len(layer_ids))
			  ] + [f'gram_{i}' for i in range(len(layer_ids))]

	def make_features(self, x, clone=False):
		self.m_feat(x)
		return [(o.clone() if clone else o) for o in self.hooks.stored]
	
	def forward(self, input, target):
		out_feat = self.make_features(target, clone=True)
		in_feat = self.make_features(input)
		self.feat_losses = [base_loss(input,target)]
		self.feat_losses += [base_loss(f_in, f_out)*w
							 for f_in, f_out, w in zip(in_feat, out_feat, self.wgts)]
		self.feat_losses += [base_loss(gram_matrix(f_in), gram_matrix(f_out))*w**2 * 5e3
							 for f_in, f_out, w in zip(in_feat, out_feat, self.wgts)]
		self.metrics = dict(zip(self.metric_names, self.feat_losses))
		return sum(self.feat_losses)
	
	def __del__(self): self.hooks.remove()

feat_loss = FeatureLoss(vgg_m, blocks[2:5], [5,15,2])
```

I trained the model on [google colab’s free gpus](https://colab.research.google.com/). They’re a great resource and I can’t believe they are free. 



# Training

The interesting thing that fastai [recommends](https://www.youtube.com/watch?v=9spwoDYwW_I) is increasing the size of your images gradually. 

So at first you train on small size images, then you upscale your images and retrain on the larger images. It saves you a lot of time. Pretty smart. 

First we'll train on images of size 128x128. Because the images are so small I can up the batch size to 64. 

```python
data = get_data(bs=64,size=128)

learn = None
gc.collect()
wd = 1e-3
learn = unet_learner(data, arch, wd=wd, loss_func=feat_loss, callback_fns=LossMetrics,blur=True, norm_type=NormType.Weight)
```

I picked a learning rate of `1e-2` for this. I wanted something aggressive, but still on the safe side of explosion. This has been [shown](https://sgugger.github.io/how-do-you-find-a-good-learning-rate.html) to be very useful.

```python

lr = 1e-2

def do_fit(save_name, lrs=slice(lr), pct_start=0.9, cycles = 10):
	learn.fit_one_cycle(cycles, lrs, pct_start=pct_start)
	learn.save(save_name)
	learn.show_results(rows=2, imgsize=7)

do_fit('1a', slice(lr))
``` 

![alt text](/images/star_wars/train_loss.png)

The network will print the results during training. The input is on the left, the prediction in the middle and the target on the right. The results look very promising!

![alt text](/images/star_wars/first_train.png)

I resized and trained again. And again. Every time I make the resize slightly larger than it was previously. I moved from 128x128 to 480x480 to the original size of the video frames. 

```python
data = get_data(bs=1)

learn.data = data
learn.freeze()
gc.collect()

learn.load('2b')

lr = 1e-6

do_fit('3b', slice(lr, 1e-3), pct_start=0.3, cycles = 2)
```

![alt text](/images/star_wars/final_train.png) 

This was the final train. For this I used `pct_start = 0.3`. I wanted to learning rate to reduce 70% of the time during training. I prefer a lower learning rate when fine tuning models. The results from this piece of training look really good. 

![alt text](/images/star_wars/final_results.png) 


# Inference: Applying to Star Wars

Once this network had trained, I ran inference. This was more involved than I originally thought. 

I had to download the Star Wars deleted scenes (using [youtube-dl](https://github.com/ytdl-org/youtube-dl)) and then extract all the frames in this video. I extracted the frames using the same method previously. 

![alt text](/images/star_wars/star_wars_frame.png) 


Then I had to run inference from the learner on each individual frame of the video. That takes a long time. 

```python
# learn = load_learner("/content/drive/My Drive/video_restorer/")
# learn = load_learner("/content/drive/My Drive/video_restorer3/")
learn = load_learner("/content/drive/My Drive/video_restorer4/")

def run_inference_images(file, dest):

  img = open_image(file)
  p,img_hr,b = learn.predict(img)

  # Image(img_hr).save(dest)
  # plt.figure(figsize=(25,25))
  Image(img_hr).show(figsize=(25,25))
  plt.savefig(dest, bbox_inches = 'tight', pad_inches = 0)
  plt.close()

def scale_to_square(orig, targ, dest):
  # a simple stretch to fit a square really makes a big difference in rendering quality/consistency.
  # I've tried padding to the square as well (reflect, symetric, constant, etc).  Not as good!
  targ_sz = (targ, targ)
  orig = orig.resize(targ_sz, resample=PIL.Image.BILINEAR)
  orig.save(dest)
  return dest

def unsquare(image, orig, dest):
  targ_sz = orig.size
  image = image.resize(targ_sz, resample=PIL.Image.BILINEAR)
  image.save(dest)
  return dest


def adjust_brightness(img, factor, dest):

  enhancer = PIL.ImageEnhance.Brightness(img)
  img = enhancer.enhance(factor)
  os.remove(dest)
  img.save(dest)
  return img

```

I added some hacks here. 

First, I added a render factor. This was taken from [Deoldify](https://github.com/jantic/DeOldify). The idea is that I downscale the image and convert it to a square. Then I run inference on that image. The model is more receptive to images that are square shaped. This has been [shown](https://github.com/jantic/DeOldify#stuff-that-should-probably-be-in-a-paper) to reduce 'glitches' considerably. 

After running inference on the square shaped image I convert it back to its original shape. I found this to reduce glitches and generally result in a smoother video output. I set the `render_factor` to `40`, although it can be higher if we want higher res output. I may need a larger RAM for that though. 

Second, I adjust brightness. This isn't really a hack. Seems like more of a mistake that I'm correctly manually. For some reason the model inference results in images that are very low in brightness. 

I suspect it's something to with the `softlight` filter we used for ffmpeg earlier. But I'm having to manually correct that here. I'll need to look into this further.  

Third, I'm using matplotlib's save functionality. I found fastai's [save image](https://docs.fast.ai/vision.image.html#Image.save) functionality to give me very weird results (luke clothes were fluroscent blue and red). But strangely matplotlib's save functionality gives me okay results. I'll need to look into this. I suspect that I may be loosing quality on the image because I'm using matplotlib's `savefig` functionality. 

```python
import PIL
import glob
import os
from tqdm.notebook import tqdm

render_factor = 40

if os.path.exists('imagepaths.txt'):
  os.remove('imagepaths.txt')

!rm -Rf seinfeld_inference/high_res/
!mkdir seinfeld_inference/high_res/

def write_to_txt(dest):
  file = open("imagepaths.txt", "a") 
  write = "file '" + dest.strip() + "'" + "\n"
  file.write(write) 
  file.close() 

files = sorted(glob.glob('seinfeld_inference/images/*.*g'), key = lambda x: int(os.path.basename(x).split('.')[0]))
files = files[300:]

for i in tqdm(range(1000)):

  # file = random.choice(files)
  file = files[i]
  dest = 'seinfeld_inference/high_res/'+os.path.basename(file)

  # scale to square
  new_path = scale_to_square(PIL.Image.open(file), render_factor*16, dest = dest.split('.')[0]+'_square.jpg')

  # run inference
  run_inference_images(new_path, dest)

  # unsquare
  dest = unsquare(PIL.Image.open(dest), PIL.Image.open(file), dest = dest.split('.')[0]+'_unsquared.jpg')

  # write to txt
  write_to_txt(dest)

  # increase brightness
  adjust_brightness(PIL.Image.open(dest),factor = 1.75, dest=dest)

```

Here's some of the outputs from the model.

![alt text](/images/star_wars/model_inference.png) 
![alt text](/images/star_wars/model_inference2.png)
![alt text](/images/star_wars/model_inference3.png)


Then I had to stitch all these frames together to create a video. To do that I initially used ffmpeg but I ended up overloading my RAM. Instead I used opencv2's `VideoWriter`.

```python
import cv2
import numpy as np
import os

clean = 'seinfeld_inference.mp4'
cap = cv2.VideoCapture(clean)
clean_fps = cap.get(cv2.CAP_PROP_FPS)
print(clean_fps)

pathOut = 'video.mp4'
try:
  os.remove(pathOut)
except:
  pass

fps = clean_fps
frame_array = []

files = sorted(glob.glob('seinfeld_inference/high_res/*_unsquared.*g'), key = lambda x: int(os.path.basename(x).split('.')[0].split('_')[0]))
print(len(files))

for i in range(len(files)):
	filename=files[i]
	# reading each files
	img = cv2.imread(filename)
	height, width, layers = img.shape
	size = (width,height)
	
	# inserting the frames into an image array
	frame_array.append(img)
out = cv2.VideoWriter(pathOut,cv2.VideoWriter_fourcc(*'MP4V'), fps, size)
for i in range(len(frame_array)):
	# writing to a image array
	out.write(frame_array[i])
out.release()
```

 Here is the final output.

{% include youtubePlayer.html id="2BcoLsuJMP0" %}

And the original video

{% include youtubePlayer.html id="f00IkrWvur4?start=159" %}


# Improvements

1) As you can see there is room for improvement. The sky needs a bit more work. But I like the vibrancy of the background. That is an interesting (and completely unplanned) effect. The goal was to remove the ‘cue marks’ (annoying black specs) from the video. I think its done okay in that respect - but there's still more to do.

I like how the network has intensified the sun though. It completely changes the the scene between Luke and Biggs when Biggs says he's joining the rebellion. 

![alt text](/images/star_wars/a_new_sun.png)
![alt text](/images/star_wars/no_sun.png) 

2) There's a weird horizontal bar line that shows up around the ``22`` second mark. I didn't add any horizontal bars in the training set so it's completely understandable that the network didn't remove that at all. But in the future I'll need to add more horizontal bars to my training set to fix these.  


3) I’m also thinking of doing more super-resolution on the video. It would be nice to show a young Luke Skywalker in high quality. To do that I could resize the images before training further. I've already downscaled the image, but potentially I could downscale it further. 

Alternatively, to achieve superres I could potentially use a ready-made upscaler such as [VapourSynth](https://github.com/AlphaAtlas/VapourSynth-Super-Resolution-Helper). This is probably the best option as the original video is already in poor quality.

4) Inference is also an issue. It tends to overload memory and crash. The result is that `42` seconds is the longest I could get for this video. I'm not completely sure how to solve this problem. But I'll need to solve it if I'm going to be using this further.

So much to do!


