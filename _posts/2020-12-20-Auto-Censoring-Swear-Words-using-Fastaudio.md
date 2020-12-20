---
layout: post
title: Auto Censoring Swear Words in Fastaudio
---

![Alt Text](https://media0.giphy.com/media/gXiyq2FNRFnPy/source.gif)

Can you automatically bleep someone on TV when they swear?

Lately I've been watching a lot of comedy clips on youtube. They're usually uncensored- unless it's on [comedy central](https://www.reddit.com/r/IASIP/comments/18s4kn/why_the_hell_does_comedy_central_have_to_censor/). 

![Alt Text](https://i.ytimg.com/vi/edwvY6xBo4Y/maxresdefault.jpg)

The clips on comedy central got me thinking. Presumably somone has to [manually](https://www.youtube.com/watch?v=PEU9DXkiZzY) go through the video and bleep out the swear words. Why couldn't we do this using deep learning?

It's an interesting idea. And I stupidly thought it would be easy. It did not turn out to be easy at all.

Audio classification has been getting a lot of [attention](https://www.kaggle.com/c/birdsong-recognition/overview) on kaggle. But so far it's been focused on short audio samples (less than a few seconds). I couldn't find a lot of information out there on longer audio samples. 

So in this article I'll describe my initial attempt at this idea. I'll be borrowing ideas and code from a [pytorch tutorial](https://pytorch.org/tutorials/intermediate/speech_command_recognition_with_torchaudio.html) and the [fastaudio](https://github.com/fastaudio/fastaudio) repository. 


# Install

First we'll need to install a few things


```
!pip install -q torch==1.7.0+cu101 torchvision==0.8.1+cu101 torchaudio==0.7.0 -f https://download.pytorch.org/whl/torch_stable.html

!pip install -q -U fastai fastcore pydub ffmpy
```

I really like `ipython-autotime`. It automatically lets me know how long cells take to run. So there's no need to run a `%%time` in each cell. 

```
# install ipython automtime.
!pip install ipython-autotime
%load_ext autotime
```

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchaudio
import matplotlib.pyplot as plt
import IPython.display as ipd
from tqdm.notebook import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

```

# Creating the Dataset

Now we can create the dataset.

After a lot of searching I managed to find a swear word dataset that was somewhat suitable for my purposes. I'll be using a dataset from 'the abuse project' which is available on [github](https://github.com/theabuseproject/tapad).


```
!git clone https://github.com/theabuseproject/tapad swear_words
```


```python
import glob
from pydub import AudioSegment
from pathlib import Path
import os
from fastcore.parallel import parallel
import mulitprocessing as mp

# use parallel processing in fastai to speed things up

def convert_mp3_to_wav(file):
  parent = Path(file).parent.name
  file_name = Path(file).stem + '.wav'
  mp3 = AudioSegment.from_mp3(file)
  output = Path(dir).parents[1]/parent/file_name
  # print(output)
  os.remove(file)
  if Path(output).exists(): os.remove(output)
  mp3.export(output, format="wav")

dir = 'swear_words/audio/en*/*.mp3' # only english swear words
files = glob.glob(dir)
parallel(convert_mp3_to_wav, files, n_workers = mp.cpu_count())
```

Next I found a dataset that had "clean" audio. By clean I mean audio without swear words in it. I'll be using the [speech commands dataset](https://www.tensorflow.org/datasets/catalog/speech_commands) from google.


# Organise Data


As you can imagine there's a lot of data prep involved. 

We'll combine the audio from swear word dataset with the audio from the speech commands dataset. Then we'll group them into `train` and `test` folders. 

To make things really easy, we'll also create a `train_df` dataframe and a `test_df` dataframe. This will contain the name of each audio file and swear word tracker. A `1` means that this file contains a swear word. A `0` means this file does not have a swear word.


The following code accomplishes all of that


```python
import pandas as pd
import shutil
import multiprocessing as mp # https://cslocumwx.github.io/blog/2015/02/23/python-multiprocessing/

clean_words = glob.glob('SpeechCommands/speech_commands_v0.02/*/*.wav')

swear_words = glob.glob(dir.replace('mp3', 'wav')); len(swear_words)

def get_label(filepath):
  if 'speech_commands' in str(filepath): # Todo: get rid of string matching solution. Something better?
    # print(f'{filepath} contains no swear words')
    return 0 # contains no swear words
  else:
    # print(f'{filepath} contains swear words')
    return 1 # contains swear words

def get_name(filepath):
  if 'speech_commands' in str(filepath):
    name = str(Path(filepath).parent.name)
  else:
    name = str(Path(filepath).stem)
  return name

def _get_train_test(data, split = 0.8):
  random.shuffle(data)
  train_length = int(len(data) * 0.8)
  train_data = data[:train_length]
  test_data = data[train_length:]
  return train_data, test_data

def _delete_recreate(text): 
  if Path(text).exists(): shutil.rmtree(text)
  os.mkdir(text)

def _get_output_name(file, dir):
  file_name = get_name(file) + str(Path(file).suffix)
  output = Path(dir)/file_name
  return output

def _copy_across(file, dir):
  shutil.copyfile(file, _get_output_name(file, dir))

def _copy_across_and_create_df(dir, data):
  print(f'preparing {dir} folder...')
  df = pd.DataFrame({'file_name': [_get_output_name(i, dir) for i in data], \
                      'target': [get_label(i) for i in data]})
  # if resample: parallel(_resample, data, dir = dir, sr = 16000, n_workers = mp.cpu_count()) 
  parallel(_copy_across, data, dir = dir, n_workers = mp.cpu_count()) # use parallel processing to speed things up
  return df
```

Now we'll use the helper functions to create the train and test datasets as well as the associated dataframes. 

```python
def create_train_test(split = 0.8):

  if Path('train.csv').exists() and Path('test.csv').exists(): 
    return pd.read_csv('train.csv'), pd.read_csv('test.csv')

  _delete_recreate('test')
  _delete_recreate('train')

  train_data, test_data = _get_train_test(swear_words + clean_words, split)

  train_df = _copy_across_and_create_df('train', train_data)
  test_df = _copy_across_and_create_df('test', test_data)

  return train_df, test_df

train_df, test_df = create_train_test(0.8)
train_df.head()

```

<img src="/images/auto_censoring/train_df_head.png" alt="img" width="400"/>


I've used multiprocessing here to really speed things up. The `parallel` function from [`fastcore`](https://fastcore.fast.ai/parallel.html#parallel) is incredibly useful and saves me a lot of time.



# Listen to Audio

Now we'll listen the a random sample of audio. Just to check we're on track.


```python
def listen_random_audio(df):
  i = random.randrange(0, df.shape[0])
  file = df.iloc[i]['file_name']
  print(df.iloc[i]['target'])
  print(file)
  w, sr = torchaudio.load(file)
  return ipd.Audio(w.numpy(), rate = sr)
listen_random_audio(train_df)
```
{% include embed-audio.html src="/assets/audio/down.wav" %}
{% include embed-audio.html src="/assets/audio/left.wav" %}


# Fastaudio

Fastaudio is a cool library for fastai. We'll try using fastaudio to build our swear word classifier. 

```
# fastai audio
!pip install -q git+https://github.com/fastaudio/fastaudio.git --upgrade
```

We'll need a few helper functions

```python
from fastai.vision.all import *
from fastaudio.core.all import *
from fastaudio.augment.all import *
from fastaudio.ci import skip_if_ci

# Helper function to split the data
def CrossValidationSplitter(col='fold', fold=1):
    "Split `items` (supposed to be a dataframe) by fold in `col`"
    def _inner(o):
        assert isinstance(o, pd.DataFrame), "ColSplitter only works when your items are a pandas DataFrame"
        col_values = o.iloc[:,col] if isinstance(col, int) else o[col]
        valid_idx = (col_values == fold).values.astype('bool')
        return IndexSplitter(mask2idxs(valid_idx))(o)
    return _inner

# Create folds column
def create_folds(iter, df):
  jump = df.shape[0]//n_folds
  if (iter == 0): return [1]
  return [iter]*jump + create_folds(iter - 1, df)
```

Now we can create the cross validation folds. We'll be using 5 folds. 

```python
n_folds = 5
train_df['fold'] = create_folds(n_folds, train_df)
train_df.head()
```

We'll extract the BasicMelSpectrogram with `512` fourier transforms. This seems to be the default config. We'll go with that for now.


```python
cfg = AudioConfig.BasicMelSpectrogram(n_fft=512, n_mels = 32)
a2s = AudioToSpec.from_cfg(cfg)
```



We'll resample all our files so that they are all 8kHZ. This should be good for [voice audio](https://forums.fast.ai/t/fastai-v2-audio/53535/99).

We'll also resize the signal so that all our audio tensors are the same size.

In addition, I think that converting sound from stereo to mono may be a [good idea](https://pytorch.org/tutorials/intermediate/speech_command_recognition_with_torchaudio.html#formatting-the-data). 


```python
item_tfms = [ResizeSignal(500), Resample(8000), DownmixMono()] # https://forums.fast.ai/t/fastai-v2-audio/53535/99
batch_tfms = [a2s]
```


In fastaudio it's important to distinguish between `item_tfms` and `batch_tfms`. I made the mistake of confusing the two and got stuck on debugging for hours. 

The `BasicMelSpectrogram` is part of the batch transform. 
But `ResizeSignal`, `Resample` and `DownmixMono` are part of the item transforms. You want to resize, resample and downmix each individual item before it is part of the batch. Don't make the mistake I did!



```python
# Note: important to distinguish between item tfms and batch tfms
# I made the mistake of confusing the two and got stuck on debugging for a few hours lol

# Note: doesn't work on GPU as yet
auds = DataBlock(blocks = (AudioBlock, CategoryBlock),  
                 get_x = ColReader("file_name"), 
                 batch_tfms = batch_tfms,
                 item_tfms = item_tfms,
                 splitter = CrossValidationSplitter(col = 'fold', fold = 2),
                 get_y = ColReader("target"))

dbunch = auds.dataloaders(train_df, bs=bs)
```

We can visualise a batch pretty easily in fastaudio. 

```python
dbunch.show_batch(figsize=(10, 5))
```
![alt text](/images/auto_censoring/dbunch_show_batch.png)


We'll be using transfer learning from a `resnet18` model. Transfer learning has proved to be very [useful](https://builtin.com/data-science/transfer-learning#:~:text=Transfer%20learning%20has%20several%20benefits,needing%20a%20lot%20of%20data.).

```python
learn = cnn_learner(dbunch, 
            models.resnet18,
            config=cnn_config(n_in=1), #<- Only audio specific modification here
            # config={"n_in":1},
            # loss_func=CrossEntropyLossFlat(),
            metrics=[F1Score()])
```

<!-- Here's a summary of the model we'll be using -->

Next we'll run the training loop with a `SaveModelCallback`. This will allow us to save the best model automatically. We'll be tracking f1 score since our classes are imbalanced.


```python
from fastai.callback.tracker import SaveModelCallback
@skip_if_ci
def run_learner():
    # epochs are a bit longer due to the chosen melspectrogram settings
    learn.fine_tune(epochs = 0, cbs = [SaveModelCallback(monitor='f1_score', fname='stage-1')])

# We only validate the model when running in CI
run_learner()
```

![alt text](/images/auto_censoring/placeholder_training_shot.png)

The f1 score looks pretty good!

```python
learn.load("stage-1")
learn.unfreeze()
```




# Prediction on long-form audio sample
Finally we'll run our model on a longer form audio sample.

I'll be using a [clip](https://www.youtube.com/watch?v=I21ANMLvntQ&ab_channel=NetflixIsAJoke) from Tom Segura's comedy special. This contains a lot of swear words.

I'll be using parallel processing again here to really speed things up. 

```
!pip install youtube-dl
!youtube-dl --extract-audio --audio-format wav https://www.youtube.com/watch?v=I21ANMLvntQ&ab_channel=NetflixIsAJoke
```


First we'll split the audio files into smaller segments. I've split things here into segments of about 0.2 seconds. That should be long enough to catch a swear word but short enough to not overwhelm the classifier. 


```python
import shutil
from pydub import AudioSegment
import os

def _get_length_audio_in_ms(file):
  audio = AudioSegment.from_wav(file)
  return int(audio.duration_seconds * 1000)

def _splitter(i, file, folder, increment):
  t1 = i 
  t2 = i + increment
  output = Path(f'{folder}/audio_{i}{Path(file).suffix}')
  audio = AudioSegment.from_wav(file)
  audio[t1:t2].export(output, format=f'{Path(file).suffix[1:]}')
  
def split_audio_file(file, folder = 'output', increment = 1000):

  print(f'splitting audio files...')
  folder = Path(folder)
  if folder.exists(): 
    shutil.rmtree(folder)
  os.mkdir(folder)

  length_ms = _get_length_audio_in_ms(file)
  parallel(_splitter, range(0, length_ms, increment), file = file, folder = folder, increment = increment)
  # for i in range(0, length_miliseconds, increment):
  #   _splitter(i, file, folder, increment)
    
  files = glob.glob(f'{folder}/audio*{Path(file).suffix}')
  files = sorted(files, key = lambda x: int(str(Path(x).stem).split('_')[-1]))
  return files

```


```python
seconds = 0.2
miliseconds = int(seconds * 1000)
files = split_audio_file('/content/Best Of - Tom Segura _ Netflix Is A Joke-I21ANMLvntQ.wav', 'output', miliseconds)

```
We can listen to the audio samples to check we're on track.

```python
def play_audio(file):
  w, sr = torchaudio.load(file)
  # print(w.shape)
  print(file)
  return ipd.Audio(w.numpy(), rate=sr)
play_audio(files[0])
```
{% include embed-audio.html src="/assets/audio/audio_0.wav" %}
{% include embed-audio.html src="/assets/audio/audio_1000.wav" %}



Next we'll download a censor audio sound. I'm using [this one](https://www.youtube.com/watch?v=RPfCZhvj1Ng) from youtube.


```
!youtube-dl --extract-audio --audio-format wav https://www.youtube.com/watch?v=RPfCZhvj1Ng
```



Now we'll get the model to run a prediction on each audio file. If the model believes that the audio file contains a swear word then it will automatically censor it using the audio sound we downloaded from youtube.




```python

import ffmpy
from tqdm.notebook import trange

CENSOR_FILE = 'Censor Beep Sound Effect-RPfCZhvj1Ng.wav'
PREDICTION_FILE = 'Best Of - Tom Segura _ Netflix Is A Joke-I21ANMLvntQ.wav'

def bleep_out_swear_word(file):
  # return censored file which is the same length as original file
  ms = _get_length_audio_in_ms(file)
  output = _get_output_name(file, dir = 'bleeped_out')
  ffmpy.FFmpeg(inputs={str(CENSOR_FILE):''},\
               outputs={str(output): "-y -ss 00:00:00 -to 00:00:00."+str(ms)+' -c copy'}).run() 
  return output

def _stitch_files_helper(file, output):
  ffmpy.FFmpeg(inputs={str(file):"-y -loglevel warning"},\
               outputs={str(output): "-filter_complex amix=inputs=1:duration=first:dropout_transition=0"}).run()
  # !ffmpeg -loglevel warning -y -i output/audio200.wav -i output/audio400.wav -i output/audio600.wav -i output/audio800.wav -i output/audio0.wav -filter_complex [0:0][1:0][2:0][3:0][4:0]concat=n=1727:v=0:a=1[out] -map [out] output.wav

def _stitch_files_together(files, output_name = 'output.wav'):
  print(f'stitching files together...')
  # parallel(_stitch_files_helper, files[1:], output = output_name, n_workers = mp.cpu_count())

  # https://superuser.com/questions/587511/concatenate-multiple-wav-files-using-single-command-without-extra-file
  n = len(files)
  t = ''.join(['['+str(i)+':0]' for i in range(n)])
  ffmpy.FFmpeg(inputs={str(files[n-1]): "-y "+ " ".join(['-i ' + str(i) for i in files[:n-1]])},\
               outputs={str('output.wav'): "-filter_complex '"+str(t)+"concat=n="+str(n)+\
               ":v=0:a=1[out]' -map '[out]'"}).run()
    
  return output_name

def predict_long_form(file, seconds = 0.2, folder = 'output'):
  _delete_recreate('bleeped_out')
  miliseconds = int(seconds * 1000)
  files = split_audio_file(file, Path(folder), miliseconds)
  
  for i in trange(len(files)):
    pred = learn.predict(files[i])[0]
    print(f'swear word identified')
    if (pred == 1): files[i] = bleep_out_swear_word(files[i])

  output_name = _stitch_files_together(files)
  return output_name

output = predict_long_form(PREDICTION_FILE, seconds = 0.2)
```


Cool let's listen and hear how it went. I ran into OOM issues when trying to play the entire file on jupyter notebook so I'll just embed a sample here. I'll put the full file on github. 


```python
def play_sample_audio(file, duration_min = 0, duration_second = 20):
  new_name = Path(file).stem + '_sample' + Path(file).suffix
  ffmpy.FFmpeg(inputs={str(file): "-y "},\
               outputs={str(new_name): "-ss 00:00:00 -to 00:"+str(duration_min)+":"+str(duration_second)}).run()
  # !ffmpeg -y -loglevel warning -i $file -ss 00:00:00 -to 00:$duration_min:$duration_second -c copy $new_name
  return play_audio(new_name)
play_sample_audio('output.wav', duration_second = 30)
```

<!-- {% include embed-audio.html src="/assets/audio/output_sample.wav" %} -->


The full jupyter notebook can be found on [Github](https://github.com/spiyer99/spiyer99.github.io/blob/master/nbs/swear_word_detection.ipynb)

