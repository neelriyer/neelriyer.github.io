---
layout: post
title: Simpsonize Yourself using CycleGAN
---

![alt text](/images/cyclegan_simpsonify/real_face_8244_fake.png)
![alt text](/images/cyclegan_simpsonify/real_face_8244_real.png)


![alt text](/images/cyclegan_simpsonify/real_face_9038_fake.png)
![alt text](/images/cyclegan_simpsonify/real_face_9038_real.png)


<!-- <table><tr><td><img src='https://drive.google.com/uc?id=1jmpktn6Jj9ia_bKpkNzU3A-vtEjRrcpc'></td><td><img src='https://drive.google.com/uc?id=1_t_Gdq8FxdhkF7QVhP_RxCDvO9a-CLjB'></td></tr></table>


<table><tr><td><img src='https://drive.google.com/uc?id=1pTuzTcVpPWZnvEtmd4FOg6vdtQ0zIWQh'></td><td><img src='https://drive.google.com/uc?id=1SV5vLt-KrXRmAesBagivP6OlzsFhbsyO'></td></tr></table> -->


[Cyclegan](https://arxiv.org/abs/1703.10593) is a framework that is capable of image to image translatation. It's been applied in some really interesting cases. Such as converting [horses to zebras](https://camo.githubusercontent.com/69cbc0371777fba5d251a564e2f8a8f38d1bf43f/68747470733a2f2f6a756e79616e7a2e6769746875622e696f2f4379636c6547414e2f696d616765732f7465617365725f686967685f7265732e6a7067) (and back again) and photos of the winter to photos of the summer. 

I thought this could be potentially applied to The Simpsons. I was inspired by sites like [turnedyellow](https://turnedyellow.com/) and [makemeyellow](https://makemeyellow.photos/). 

The idea is that you upload a photo of your face and Cyclegan would translate that into a Simpsons Character. 

In this article I describe the workflow required to  'Simpsonise' yourself using Cyclegan. It's worth noting that the [paper](https://arxiv.org/pdf/1703.10593.pdf) explicitly mentions that large geometric changes are usually unsuccessful. 

But I'm going to attempt this anyway.

# Install

First we need to install Cyclegan

```
!git clone https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
import os
os.chdir('pytorch-CycleGAN-and-pix2pix/')
!pip install -r requirements.txt
```

# Datasets

Creating the dataset is harder than you would initially think.

To create this dataset I needed to find close up shots of Simpsons characters and close up shots of regular people.

### Download Simpsons Faceset

Initially my thinking was to scrape images from google images. Unfortunately to do that it looks like you need a [developer key](https://pypi.org/project/Google-Images-Search/) from google console.

So instead I scraped images from [Bing](https://github.com/gurugaurav/bing_image_downloader). 

This worked to an extent. But it took *so long* to download all the images. And after I looked at the images I noticed that some of them didn't include any faces at all.

Thankfully, I stumbled across a [faceset](https://www.kaggle.com/kostastokis/simpsons-faces) on kaggle that had everything I needed. It contains Simpsons faces extracted from a few seasons. Each image is 200x200 pixels and contains one face. 

This faceset will be stored in the `trainA` folder.


```python
def create_training_dataset_simpson_faces(download_dir):

  %cd $download_dir

  # download dataset and unzip
  !kaggle datasets download kostastokis/simpsons-faces --force
  !unzip \*.zip
  !rm *.zip
  !cp -a $download_dir/cropped/. $download_dir

  # remove unnecessary folders
  !rm -Rf $download_dir/cropped
  !rm -Rf $download_dir/simplified

  # go back to orig directory
  %cd /content/pytorch-CycleGAN-and-pix2pix

create_training_dataset_simpson_faces(TRAIN_A)

```


### Download Real Faceset


To create the faceset of real faces I got a little bit experimental.

[Will Kwan](https://www.youtube.com/watch?v=pctzpu_wJyE) recently using [stylegan2](https://github.com/NVlabs/stylegan2) to generate a dataset in one of his recent videos. It seemed to work fairly well for him. So I thought I could do the same thing.

Here's some example faces taken from Nvidia's stylegan2 github repository. As you can see the output from this GAN is fairly photorealistic. 

<div>
<img src="https://miro.medium.com/max/2636/1*WR5OApYceVhZTNhO33csZw.png" width="500" height = "500"/>
</div>

This could turn out far better than scraping the internet for faces. I could create as many faces as I needed for my model. I wouldn't have to download some cumbersome `zip` file.  

This faceset will be stored in the `trainB` folder


#### Stylegan Create Faceset


```python
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm

def create_training_dataset_real_faces_stylegan(download_dir):

  # create in batches of 100
  # reduces RAM requirements

  counter = 0
  pbar = tqdm(total = LIMIT)

  while counter < LIMIT:

    seeds = np.random.randint(10000000, size=100)
    imgs = generate_images_from_seeds(seeds, 0.7)

    for img in imgs:
      img.save(download_dir/'real_face_{}.jpg'.format(counter), 'JPEG', quality=100)
      counter+=1
      pbar.update(1)
    del imgs

create_training_dataset_real_faces_stylegan(TRAIN_B)
```


### Train Test Split

Next we'll need to split the data into training and testing. `testB` will contain real faces that we want to covnert into Simpsons characters. `testA` will contain simpsons characters that we want to convert into real people.



```python
# move images to a new folder
# `images` is the existing image directory: 
# `new_dir` is the path that the images will be moved to
# `files_limit` is the limit of files that will be moved
def move_all_images_to_new_folder(images, new_dir, files_limit = None):
  files = glob.glob(str(images/'*.*g'))

  if(files_limit is not None):
    files = files[:files_limit]

  for file in files: shutil.move(file, new_dir/os.path.basename(file))

move_all_images_to_new_folder(TRAIN_A, new_dir = TEST_A, files_limit = int(min(LIMIT*0.1, 25)))
move_all_images_to_new_folder(TRAIN_B, new_dir = TEST_B, files_limit = int(min(LIMIT*0.1, 25)))
```



### See our Training and Testing Data

Let's see the images we're working with. 



```python
import PIL
import random

def plot_from_image_path(path, title):

  all_imgs = glob.glob(str(path/'*.*g'))

  print(f'{len(all_imgs)} imgs in {title} directory')

  img_path = random.choice(all_imgs)
  img = PIL.Image.open(img_path)
  plt.imshow(img)
  plt.title(title)
  plt.show()

plot_from_image_path(TRAIN_A, 'TRAIN_A')
plot_from_image_path(TRAIN_B, 'TRAIN_B')

plot_from_image_path(TEST_A, 'TEST_A')
plot_from_image_path(TEST_B, 'TEST_B')

```

![alt text](/images/cyclegan_simpsonify/trainA.png)
![alt text](/images/cyclegan_simpsonify/trainB.png)
![alt text](/images/cyclegan_simpsonify/testA.png)
![alt text](/images/cyclegan_simpsonify/testB.png)

Everything looks pretty good!

# Creating the model

Now we can create the model. I've made a few adjustments to the [existing script](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/train.py). 

```python
import gc; gc.collect()
NAME = 'person2simpson'
BATCH_SIZE = 2
```


Let's create a few helper functions. 

These functions help me copy the saved models to my google drive. It also helps run inference on the model and store the output images to google drive.

```python

import os
from pathlib import Path
from distutils.dir_util import copy_tree
import matplotlib.pyplot as plt
import random

def copy_to_drive(folder = 'cyclegan_simpsonify'):

  drive_folder = Path('/content/drive/My Drive/')/folder

  if(drive_folder.exists()):
    shutil.rmtree(drive_folder)

  shutil.copytree('/content/pytorch-CycleGAN-and-pix2pix/checkpoints/'+NAME+'/', str(drive_folder))

def get_corresponding_photo(file_path):
  return file_path.replace('fake', 'real')

def plot_results(number):

  for i in range(number):

    img_path = random.choice(glob.glob('./results/'+NAME+'/test_latest/images/*fake.*g'))
    print(img_path)
    img = plt.imread(img_path)
    plt.imshow(img)
    plt.title('fake')
    plt.show()

    print(get_corresponding_photo(img_path))
    img = plt.imread(get_corresponding_photo(img_path))
    plt.imshow(img)
    plt.title('real')
    plt.show()

def get_model(src, dst):

  # copy across model
  try:
    os.remove(dst)
  except:
    pass
  shutil.copyfile(src, dst)

def copy_from_drive(folder = 'cyclegan_simpsonify'):

  drive_folder = Path('/content/drive/My Drive/')/folder

  if(not Path('/content/pytorch-CycleGAN-and-pix2pix/checkpoints/').exists()): 
    os.mkdir('/content/pytorch-CycleGAN-and-pix2pix/checkpoints/')

  if(Path('/content/pytorch-CycleGAN-and-pix2pix/checkpoints/'+NAME+'/').exists()): 
    shutil.rmtree('/content/pytorch-CycleGAN-and-pix2pix/checkpoints/'+NAME+'/')

  shutil.copytree(str(drive_folder), '/content/pytorch-CycleGAN-and-pix2pix/checkpoints/'+NAME+'/')

def test_model (number_results = 5, direction = 'BtoA', src = None, dst = None):

  # delete results folder and recrete
  shutil.rmtree('./results')
  os.mkdir('./results')

  # get appropriate model
  if (src is None): src = './checkpoints/'+NAME+'/latest_net_G_'+direction.split('to')[-1]+'.pth'
  if (dst is None): dst = './checkpoints/'+NAME+'/latest_net_G.pth'

  get_model(src, dst)

  if (direction == 'BtoA'):
    test = TEST_B
  else:
    test = TEST_A
  
  cmd = 'python test.py --dataroot '+str(test)+' --name '+str(NAME)+' --model test --no_dropout'
  os.system(cmd)
  plot_results(number_results)

```

Let's create the options for training. This part is fairly long.


```python
import time
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
from util.visualizer import Visualizer
import shutil
import os
from pathlib import Path
from tqdm.notebook import tqdm

options_list = ['--name', NAME,\
				'--dataroot', TRAIN_A.parent,\
				'--batch_size', BATCH_SIZE,\
				'--checkpoints_dir', './checkpoints',\
				'--lr', 2e-4,\
				'--n_epochs', EPOCHS,\
				'--n_epochs_decay', EPOCHS//2,\
				'--name', NAME]

opt = TrainOptions().parse(options_list)

```


First we create the dataset using the options we specified earlier. 


```python
dataset = create_dataset(opt)
dataset_size = len(dataset)
print('The number of training images = %d' % dataset_size)
```

Then we create the model and run the setup call.


```python
model = create_model(opt)
model.setup(opt)
visualizer = Visualizer(opt)
total_iters = 0

```

Let's see the generator from A to B. The naming convention is slightly different from the [paper](https://arxiv.org/pdf/1703.10593.pdf). In the [paper](https://arxiv.org/pdf/1703.10593.pdf) the generator was called `G`. 

In the code they refer to this mapping function as `G_A`. The meaning is still the same. 

This generator function maps from `A` to `B`. 

In our case it maps from Simpsons to Real Life.


```python
model.netG_A
```
![alt text](/images/cyclegan_simpsonify/generator_A.png)

We can see here that the model uses [Resnets](https://arxiv.org/pdf/1512.03385v1.pdf). It has several Resnet blocks. 

We have `Conv2d`, `Batchnorm`, `ReLU`, `InstanceNorm2d` and `ReflectionPad2d`. `InstanceNorm2d` and `ReflectionPad2d` are new to me.

`InstanceNorm2d`: This is very similar to [batch norm](https://arxiv.org/abs/1502.03167) but it is applied to one image at a time. 

`ReflectionPad2d`:  This will pad the tensor using the reflection of the input boundary.


Now we can look at the discriminator as well.

```python
model.netD_A 
```

![alt text](/images/cyclegan_simpsonify/discriminator.png)


The discriminator uses `LeakyReLU`, `Conv2d` and `InstanceNorm2d`. 

`LeakyReLU` is interesting. `ReLU` is an activation that adds non-linearity to the network. But what is  `LeakyReLU`? 

`ReLU` converts all negative values to `0`. Since, the gradient of `0` is `0` neurons that reach large negative effectively neuron cancel out to `0`. They effectively 'die'. This means that your network eventually stops learning. 

This effect is known as the [dying `ReLU` problem](https://datascience.stackexchange.com/questions/5706/what-is-the-dying-relu-problem-in-neural-networks). 

`LeakyReLU` aims to fix this problem. The function is as follows:



![alt text](/images/cyclegan_simpsonify/leaky_relu.png)

​	
This function essentially translates to: if a value is negative mulitply it by `negative_slope` otherwise do nothing. `negative_slope` is usually `0.01`, but you can vary it. 

So `LeakyReLU` signficantly reduces the magnitude of negative values rather than sending them to `0`. But the jury is [still out](https://www.quora.com/What-are-the-advantages-of-using-Leaky-Rectified-Linear-Units-Leaky-ReLU-over-normal-ReLU-in-deep-learning/answer/Nouroz-Rahman) on whether this really works well.


# Training

Now we can train the model over a number of epochs. I've specified `10` epochs here.

`model.optimize_parameters()` is where all the magic happens. It optimises the generators. Then it optimises the discriminators. 

The generator takes a horse and tries to generate a zebra. Then it runs the discriminator on the generated zebra and passes this generated zebra to the GAN loss function. It does the same thing for zebras. 

Then we compute the cycle consistency loss. So we take our fake zebra and try to turn it back into a horse. Then we compare this new horse to our original horse. We do the same for zebras. 

We analyse the combined loss which is comprised of the cycle consistency loss and the GAN loss.

After calculating this loss function the model calculates the gradients and updates the weights.

Let's train it!


```python

# adapted from: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/test.py

for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
    epoch_start_time = time.time()  # timer for entire epoch
    iter_data_time = time.time()    # timer for data loading per iteration
    epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch
    visualizer.reset()              # reset the visualizer: make sure it saves the results to HTML at least once every epoch
    model.update_learning_rate()    # update learning rates in the beginning of every epoch.
    
    for i, data in enumerate(dataset):  # inner loop within one epoch
        iter_start_time = time.time()  # timer for computation per iteration

        if total_iters % opt.print_freq == 0:
            t_data = iter_start_time - iter_data_time

        total_iters += opt.batch_size
        epoch_iter += opt.batch_size
        model.set_input(data)         # unpack data from dataset and apply preprocessing
        model.optimize_parameters()   # calculate loss functions, get gradients, update network weights

        if total_iters % opt.display_freq == 0:   # display images on visdom and save images to a HTML file
            save_result = total_iters % opt.update_html_freq == 0
            model.compute_visuals()
            visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

        if total_iters % opt.print_freq == 0:    # print training losses and save logging information to the disk
            losses = model.get_current_losses()
            t_comp = (time.time() - iter_start_time) / opt.batch_size
            visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)
            if opt.display_id > 0:
                visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, losses)

        if total_iters % opt.save_latest_freq == 0:   # cache our latest model every <save_latest_freq> iterations
            print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
            save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
            model.save_networks(save_suffix)

        iter_data_time = time.time()


    if epoch % opt.save_epoch_freq == 0:              # cache our model every <save_epoch_freq> epochs
        print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
        model.save_networks('latest')
        model.save_networks(epoch)
        copy_to_drive()
        test_model(1, 'AtoB')
        test_model(1, 'BtoA')

    print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))



```

# Testing

I let the model train overnight on [google colab](https://colab.research.google.com/) and copied the `.pth` model to my google drive. 

We'll copy back the weights from google drive to my local computer.

```python
test_model(10, 'BtoA')

```

![alt text](/images/cyclegan_simpsonify/real_face_818_fake.png)
![alt text](/images/cyclegan_simpsonify/real_face_818_real.png)


![alt text](/images/cyclegan_simpsonify/real_face_2224_fake.png)
![alt text](/images/cyclegan_simpsonify/real_face_2224_real.png)



![alt text](/images/cyclegan_simpsonify/real_face_7874_fake.png)
![alt text](/images/cyclegan_simpsonify/real_face_7874_real.png)


![alt text](/images/cyclegan_simpsonify/real_face_8244_fake.png)
![alt text](/images/cyclegan_simpsonify/real_face_8244_real.png)



![alt text](/images/cyclegan_simpsonify/real_face_9038_fake.png)
![alt text](/images/cyclegan_simpsonify/real_face_9038_real.png)



It's a good start. I particularly like this image:


<table><tr><td><img src='https://drive.google.com/uc?id=1pTuzTcVpPWZnvEtmd4FOg6vdtQ0zIWQh'></td><td><img src='https://drive.google.com/uc?id=1SV5vLt-KrXRmAesBagivP6OlzsFhbsyO'></td></tr></table>


But it could use some improvement, to be honest. 

Let's try running the `BtoA` cycle. So we'll convert simpsons characters into human faces.


```python
test_model(10, 'AtoB')
```

![alt text](/images/cyclegan_simpsonify/26_fake.png)
![alt text](/images/cyclegan_simpsonify/26_real.png)


![alt text](/images/cyclegan_simpsonify/5121_fake.png)
![alt text](/images/cyclegan_simpsonify/5121_real.png)


![alt text](/images/cyclegan_simpsonify/8899_fake.png)
![alt text](/images/cyclegan_simpsonify/8899_real.png)

![alt text](/images/cyclegan_simpsonify/9759_fake.png)
![alt text](/images/cyclegan_simpsonify/9759_real.png)


# Improvements

The authors of Cyclegan [noted](https://junyanz.github.io/CycleGAN/) that tasks that require geomtric changes haven't been very successful so far. I've just confirmed this.

The network seems to struggle with the large geometric shifts required to convert a simpsons chartacter to a real person (and vice-versa). I'm unsure if more training would rectify this issue. Cyclegan seems to work well with things like texture changes, color changes and photo to painting translations. 


The full jupyter notebook can be found on [Github](https://github.com/spiyer99/spiyer99.github.io/blob/master/nbs/cyclegan_simpsonify.ipynb)



