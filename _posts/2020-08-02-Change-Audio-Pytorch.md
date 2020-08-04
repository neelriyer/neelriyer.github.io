---
layout: post
title: Neural Style Transfer for Audio in Pytorch
---

![photo](https://camo.githubusercontent.com/974884c2fb949b365c3f415b3712d2cac04a35f7/68747470733a2f2f692e696d6775722e636f6d2f575771364931552e6a7067)

Neural Style transfer is really interesting. They've been some really interesting applications of style transfer. It basically aims to take the 'style' from one image and change the 'content' image to meet that style. The image above shows an example. This image has been converted to look like it was painted by Van gough.

But so far it hasn't really been applied to audio. So this week I explored the idea of applying neural style transfer to audio. To be frank, the results were less than stellar but I'm going to keep working on this in the future. 

# Build Dataset

For this exercise, I'm going to be using clips from the joe rogan podcast. I'm trying to make [Joe Rogan](https://en.wikipedia.org/wiki/Joe_Rogan), from the [Joe Rogan Experience](http://podcasts.joerogan.net/), sound like [Joey Diaz](https://en.wikipedia.org/wiki/Joey_Diaz), from the [Church of Whats Happening Now](https://www.youtube.com/channel/UCv695o3i-JmkUB7tPbtwXDA). Joe Rogan already does a pretty good [impression of joey diaz](https://www.youtube.com/watch?v=SLolljsbbFs). But I'd like to improve his impression using deep learning.

![photo](https://talentrecap.com/wp-content/uploads/2020/06/Joe-Rogan-and-Joey-Diaz-2020-2.png)

First I'm going to download the youtube videos. There's a neat trick mentioned on [github](https://github.com/ytdl-org/youtube-dl/issues/622#issuecomment-162337869) that allows you to download small segments of youtube videos. That's handy cause I don't want to download the entire video. You'll need [youtube-dl](https://github.com/ytdl-org/youtube-dl) and [ffmpeg](https://ffmpeg.org/) for this step.

```python
import os

# download youtube url using ffmpeg
# adapted from: https://github.com/ytdl-org/youtube-dl/issues/622#issuecomment-162337869
def download_from_url_ffmpeg(url, output, minute_mark = 1):

	try:
		os.remove(output)
	except:
		pass

	# cmd = 'ffmpeg -loglevel warning -ss 0 -i $(youtube-dl -f 22 --get-url https://www.youtube.com/watch?v=mMZriSvaVP8) -t 11200 -c:v copy -c:a copy react-spot.mp4'
	cmd = 'ffmpeg -loglevel warning -ss 0 -i $(youtube-dl -f bestaudio --get-url '+str(url)+') -t '+str(minute_mark*60)+' '+str(output)
	os.system(cmd)

	return os.getcwd()+'/'+output


url = 'https://www.youtube.com/watch?v=-xY_D8SMNtE'
content_audio_name = download_from_url_ffmpeg(url, 'jre.wav')
url = 'https://www.youtube.com/watch?v=-l88fMJcvWE'
style_audio_name = download_from_url_ffmpeg(url, 'joey_diaz.wav')
```


# Loss

There are two types of loss for this:

1. Content loss. Lower values for this means that the output audio sounds like joe rogan.

2. Style loss. Lower values for this means that the output audio sounds like joey diaz.

Ideally we want both content and style loss to be minimised.

## Content loss

The content loss function takes in an input matrix and a content matrix. The content matrix corresponds to joe rogan's audio. Then it returns the weighted content distance: $w_{CL}.D_C^L(X,C)$ between the input matrix $X$ and the content matrix $C$. This is [implemented](https://pytorch.org/tutorials/advanced/neural_style_tutorial.html#content-loss) as using a torch module. It can be calculated using ``nn.MSELoss``.

This implementation of content loss was largely borrowed from [here](https://ghamrouni.github.io/stn-tuto/advanced/neural_style_tutorial.html). 

```python
import torch
import torch.nn as nn

# adapted from: https://ghamrouni.github.io/stn-tuto/advanced/neural_style_tutorial.html#
class ContentLoss(nn.Module):

		def __init__(self, target, weight):
				super(ContentLoss, self).__init__()
				# we 'detach' the target content from the tree used
				self.target = target.detach() * weight
				# to dynamically compute the gradient: this is a stated value,
				# not a variable. Otherwise the forward method of the criterion
				# will throw an error.
				self.weight = weight
				self.criterion = nn.MSELoss()

		def forward(self, input):
				self.loss = self.criterion(input * self.weight, self.target)
				self.output = input
				return self.output

		def backward(self, retain_graph=True):
				self.loss.backward(retain_graph=retain_graph)
				return self.loss

```

## Style loss

When looking at the style we really just want to extract the way in which joey diaz speaks. We don't really want to extract the exact words he says. But we want to get the tone, the intonation, the inflection, etc. from his speech. For that we'll need to get the gram matrix. 

To calculate this we get the first slice in the input matrix and flatten it. Flattening this slice in the matrix removes a lot of audio information. Then we take another slice from the input matrix and flatten it. We take the dot product of the flattened matrices. 

![photo](https://www.w3resource.com/w3r_images/numpy-manipulation-ndarray-flatten-function-image-1.png)

A dot product is a measure of how similar the two matrices are. If the matrices are similar then the we'll get a really large result. If they are very different we'll get a very small result. 

So for example, let's say that the first flattened matrix corresponded with pitch. And let's say that the second flattened matrix corresponded with volume. If we get a high dot product, then it's saying that when volume is high pitch is also high. Or in other words when joey talks very loudly his voice increases in pitch. 

The dot products can give us very large numbers. We normalize them by dividing each element by the total number of elements in the matrix.  

```python

# adapted from: https://github.com/alishdipani/Neural-Style-Transfer-Audio/blob/master/NeuralStyleTransfer.py  

import torch
import torch.nn as nn

class GramMatrix(nn.Module):

	def forward(self, input):
		a, b, c = input.size()  # a=batch size(=1)
				# b=number of feature maps
				# (c,d)=dimensions of a f. map (N=c*d)
		features = input.view(a * b, c)  # resise F_XL into \hat F_XL
		G = torch.mm(features, features.t())  # compute the gram product
				# we 'normalize' the values of the gram matrix
				# by dividing by the number of element in each feature maps.
		return G.div(a * b * c)


class StyleLoss(nn.Module):

	def __init__(self, target, weight):
		super(StyleLoss, self).__init__()
		self.target = target.detach() * weight
		self.weight = weight
		self.gram = GramMatrix()
		self.criterion = nn.MSELoss()

	def forward(self, input):
		self.output = input.clone()
		self.G = self.gram(input)
		self.G.mul_(self.weight)
		self.loss = self.criterion(self.G, self.target)
		return self.output

	def backward(self,retain_graph=True):
		self.loss.backward(retain_graph=retain_graph)
		return self.loss

```


# Converting Wav to Matrix

To convert the waveform audio to a matrix that we can pass to pytorch I'll use `librosa`. Most of this code was borrowed from Dmitry Ulyanov's [github repo](https://github.com/DmitryUlyanov/neural-style-audio-tf/blob/master/neural-style-audio-tf.ipynb) and Alish Dipani's [github repo](https://github.com/alishdipani/Neural-Style-Transfer-Audio). 

We get the Short-time Fourier transform from the audio using the `librosa` library. The window size for this is `2048`, which is also the default setting. There is scope here for replacing the code with code from torchaudio. But this works for now.

```python
from torch.autograd import Variable
import librosa
import numpy as np
import torch

N_FFT=2048
def read_audio_spectum(filename):
	x, fs = librosa.load(filename)
	S = librosa.stft(x, N_FFT)
	p = np.angle(S)
	S = np.log1p(np.abs(S))  
	return S, fs

style_audio, style_sr = read_audio_spectum(style_audio_name)
content_audio, content_sr = read_audio_spectum(content_audio_name)

if(content_sr != style_sr):
	raise 'Sampling rates are not same'
	
style_audio = style_audio.reshape([1,1025,style_audio.shape[1]])
content_audio = content_audio.reshape([1,1025,style_audio.shape[1]])

if torch.cuda.is_available():
	style_float = Variable((torch.from_numpy(style_audio)).cuda())
	content_float = Variable((torch.from_numpy(content_audio)).cuda())	
else:
	style_float = Variable(torch.from_numpy(style_audio))
	content_float = Variable(torch.from_numpy(content_audio))

```

# Create CNN

This CNN is very shallow. It consists of 2 convolutions and a ReLU in between them. I originally took the CNN used [here](https://github.com/alishdipani/Neural-Style-Transfer-Audio/blob/master/NeuralStyleTransfer.py) but I've made a few changes. 

 - Firstly, I added content loss. This wasn't added before and is obviously very useful. We'd like to know how close (or far away) the audio sounds to the original content.

 - Secondly, I added a ReLU to the model. It's pretty well [established](https://stats.stackexchange.com/questions/275358/why-is-increasing-the-non-linearity-of-neural-networks-desired) that nonlinear activations are desired in a neural network. Adding a ReLU improved the model significantly.

 - Increased the number of steps. From ``2500`` to `20000`

 - Slightly deepened the network. I added a layer of `Conv1d`. After this layer style loss and content loss is calculated. This improved the model as well, but adding ReLU resulted in the largest improvement by far.

```python
import torch
import torch.nn as nn
from torch.nn import ReLU, Conv1d
import torch.optim as optim
import numpy as np 
import copy

class CNNModel(nn.Module):
	def __init__(self):
			super(CNNModel, self).__init__()
			self.cnn1 = Conv1d(in_channels=1025, out_channels=4096, kernel_size=3, stride=1, padding=1)
			self.relu = ReLU()
			self.cnn2 = Conv1d(in_channels=4096, out_channels=4096, kernel_size=3, stride=1, padding=1)

	def forward(self, x):
		out = self.cnn1(x)
		out = self.relu(out)
		out = self.cnn2(x)
		return out

cnn = CNNModel()
if torch.cuda.is_available():
	cnn = cnn.cuda()


style_weight=1000
content_weight = 2


def get_style_model_and_losses(cnn, style_float,\
															 content_float=content_float,\
															 style_weight=style_weight):
	
	cnn = copy.deepcopy(cnn)

	style_losses = []
	content_losses = []

	# create model
	model = nn.Sequential()

	# we need a gram module in order to compute style targets
	gram = GramMatrix()

	# load onto gpu  
	if torch.cuda.is_available():
		model = model.cuda()
		gram = gram.cuda()

	# add conv1
	model.add_module('conv_1', cnn.cnn1)

	# add relu
	model.add_module('relu1', cnn.relu)

	# add conv2
	model.add_module('conv_2', cnn.cnn2)

	# add style loss
	target_feature = model(style_float).clone()
	target_feature_gram = gram(target_feature)
	style_loss = StyleLoss(target_feature_gram, style_weight)
	model.add_module("style_loss_1", style_loss)
	style_losses.append(style_loss)

	# add content loss
	target = model(content_float).detach()
	content_loss = ContentLoss(target, content_weight)
	model.add_module("content_loss_1", content_loss)
	content_losses.append(content_loss)

	return model, style_losses, content_losses

```

I personally found that my loss values - particularly for style loss - were very low. So low they were almost `0`. I recitifed this by multiplying by a `style_weight` and a `content_weight`. This seems like a crude solution. But according to [fastai](https://youtu.be/xXXiC4YRGrQ?t=5798) you care about the direction of the loss and its relative size. So I think it's alright for now.

# Run style transfer

Now I'll run the style transfer. This will use the `optim.Adam` optimizer. This piece of code was taken from the pytorch tutorial for [neural style transfer](https://pytorch.org/tutorials/advanced/neural_style_tutorial.html). For each iteration of the network the style loss and content loss is calculated. In turn that is used to get the gradients. The gradients are mulitplied by the learning rates. That in turn updates the input audio matrix. In pytorch the optimizer requries a [closure](https://pytorch.org/tutorials/advanced/neural_style_tutorial.html#gradient-descent) function.

```python
import torch
import torch.nn as nn
from torch.nn import Conv2d, ReLU, AvgPool1d, MaxPool2d, Linear, Conv1d
from torch.autograd import Variable
import torch.optim as optim
import numpy as np 
import os
import torchvision.transforms as transforms

import gc; gc.collect()

input_float = content_float.clone()
#input_float = Variable(torch.randn(content_float.size())).type(torch.FloatTensor)

learning_rate_initial = 1e-4

def get_input_param_optimizer(input_float):
	input_param = nn.Parameter(input_float.data)
	# optimizer = optim.Adagrad([input_param], lr=learning_rate_initial, lr_decay=0.0001,weight_decay=0)
	optimizer = optim.Adam([input_param], lr=learning_rate_initial)
	# optimizer = optim.SGD([input_param], lr=learning_rate_initial)
	# optimizer = optim.RMSprop([input_param], lr=learning_rate_initial)
	return input_param, optimizer

num_steps= 10000


# from https://pytorch.org/tutorials/advanced/neural_style_tutorial.html
def run_style_transfer(cnn, style_float=style_float,\
											 content_float=content_float,\
											 input_float=input_float,\
											 num_steps=num_steps, style_weight=style_weight): 
	print('Building the style transfer model..')
	# model, style_losses = get_style_model_and_losses(cnn, style_float)
	model, style_losses, content_losses = get_style_model_and_losses(cnn, style_float, content_float)
	input_param, optimizer = get_input_param_optimizer(input_float)
	print('Optimizing..')
	run = [0]

	while run[0] <= num_steps:
		def closure():
						# correct the values of updated input image
			input_param.data.clamp_(0, 1)

			optimizer.zero_grad()
			model(input_param)
			style_score = 0
			content_score = 0

			for sl in style_losses:
				#print('sl is ',sl,' style loss is ',style_score)
				style_score += sl.loss

			for cl in content_losses:
				content_score += cl.loss

			style_score *= style_weight
			content_score *= content_weight

			loss = style_score + content_score
			loss.backward()

			run[0] += 1
			if run[0] % 100 == 0:
				print("run {}:".format(run))
				print('Style Loss : {:4f} Content Loss: {:4f}'.format(
										style_score.item(), content_score.item()))
				print()

			return style_score + content_score

		optimizer.step(closure)

	# ensure values are between 0 and 1
	input_param.data.clamp_(0, 1)

	return input_param.data


output = run_style_transfer(cnn, style_float=style_float, content_float=content_float, input_float=input_float)

``` 

# Reconstruct the Audio

Finally the audio needs to be reconstructed. To do that the librosa inverse short-time fourier transform can be used. 

Then we write to an audio file and use the jupyter notebook extension to play the audio in the notebook. 

```python
# taken from: https://github.com/alishdipani/Neural-Style-Transfer-Audio/blob/master/NeuralStyleTransfer.py

if torch.cuda.is_available():
	output = output.cpu()

output = output.squeeze(0)
output = output.numpy()

N_FFT=2048
a = np.zeros_like(output)
a = np.exp(output) - 1

# This code is supposed to do phase reconstruction
p = 2 * np.pi * np.random.random_sample(a.shape) - np.pi
for i in range(500):
	S = a * np.exp(1j*p)
	x = librosa.istft(S)
	p = np.angle(librosa.stft(x, N_FFT))

OUTPUT_FILENAME = 'output.wav'
librosa.output.write_wav(OUTPUT_FILENAME, x, style_sr)
Audio(OUTPUT_FILENAME)

```

The notebook for this can be found on [Github](https://github.com/spiyer99/spiyer99.github.io/blob/master/nbs/Neural%20Transfer%20of%20Audio%20in%20Pytorch.ipynb)









 