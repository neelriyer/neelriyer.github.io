---
layout: post
title:  Collaborative Filtering in Pytorch
---

###### Building a Neural Network to understand user preferences in movie choices

![alt text](/images/pytorch_recommendation/intro.jpeg)
photo from [mc.ai](https://mc.ai/deep-learning-for-collaborative-filtering-using-fastai/)

*"You’re the average of the five people spend the most time with."* - [Jim Rohn](https://en.wikipedia.org/wiki/Jim_Rohn)

Collaborative filtering is a tool that companies are increasingly using. Netflix uses it to recommend shows for us to watch. [Facebook](https://engineering.fb.com/core-data/recommending-items-to-more-than-a-billion-people/) uses it to recommend who we should be friends with. [Spotify](https://medium.com/s/story/spotifys-discover-weekly-how-machine-learning-finds-your-new-music-19a41ab76efe) uses it to recommend playlists and songs for us. It's incredibly useful in recommending products to customers. 

In this post, I construct a collaborative filtering neural network with embeddings to understand how users would feel towards certain movies. From this we can recommend movies for them to watch.

The dataset is taken from [here](http://files.grouplens.org/datasets/movielens/). This code is loosely based of the [fastai notebook](https://github.com/fastai/course-v3/blob/master/nbs/dl1/lesson4-collab.ipynb).

# Data Prep

```python
import pandas as pd
ratings = pd.read_csv('ratings.csv')
movies = pd.read_csv(path+'movies.csv')
``` 

First, let get rid of the annoyingly complex user ids. We can make do with plain old integers. They're much easier to handle.

```python
u_uniq = ratings.userId.unique()
user2idx = {o:i for i,o in enumerate(u_uniq)}
ratings.userId = ratings.userId.apply(lambda x: user2idx[x])
```

Then we'll do the same thing for movie ids as well.

```python
m_uniq = ratings.movieId.unique()
movie2idx = {o:i for i,o in enumerate(m_uniq)}
ratings.movieId = ratings.movieId.apply(lambda x: movie2idx[x])
```

We'll need to get the number of users and the number of movies. 

```python
n_users=int(ratings.userId.nunique())
n_movies=int(ratings.movieId.nunique())
```

# The Neural Net with Embeddings

First let's create some random weights. We need to call [``super().__init__()``](https://docs.python.org/2/library/functions.html#super). This allows us to avoid calling the base class explicitly. This makes the code more maintainable. 

These weights will be uniformly distributed between 0 and 0.05. The `_` operator at the end of `uniform_` denotes an inplace operation. 

```python
class EmbeddingDot(nn.Module):
	def __init__(self):
		super().__init__()
		self.u.weight.data.uniform_(0,0.05)
		self.m.weight.data.uniform_(0,0.05)
	def forward(self):
		pass
```

Next we add our Embedding matrices and latent factors. 

```python
class EmbeddingDot(nn.Module):
	def __init__(self, n_users, n_movies):
		super().__init__()
		self.u = nn.Embedding(n_users, n_factors)
		self.m = nn.Embedding(n_movies, n_factors)
		self.u.weight.data.uniform_(0,0.05)
		self.m.weight.data.uniform_(0,0.05)
		
	def forward(self, cats, conts):
		pass
```

We're creating an embedding matrix for our user ids and our movie ids. An embedding is basically an array lookup. When we mulitply our one-hot encoded user ids by our weights most calculations cancel to `0` `(0 * number = 0)`. All we're left with is a particular row in the weight matrix. That's basically [just an array lookup](https://youtu.be/CJKnDu2dxOE?t=1625).

So we can skip the matrix mulitply and we can skip the one-hot encoded. Instead we can just do an array lookup. This [reduces memory usage](https://arxiv.org/pdf/1604.06737) and speeds up the neural network relative. It also reveals the intrinsic properties of the categorical variables. This was applied in a recent [Kaggle competition](https://www.kaggle.com/c/rossmann-store-sales) and [achieved 3rd place](https://www.kaggle.com/c/rossmann-store-sales/discussion/17974).

The size of these embedding matrices will be determined by n_factors. These factors determine the number of latent factors in our dataset. 

[Latent factors](https://en.wikipedia.org/wiki/Latent_variable). are immensely useful in our network. They reduce the need for feature engineering. For example, if `User_id` `554` likes Tom cruise and `Tom cruise` appears in a movie. User `554` will probably like the movie. `Tom cruise` appearing in a movie would be a latent feature. We didn't specify it before training. It just showed up.

Finally, we'll need to add our `forward` function.

```python
class EmbeddingDot(nn.Module):
	def __init__(self, n_users, n_movies):
		super().__init__()
		self.u = nn.Embedding(n_users, n_factors)
		self.m = nn.Embedding(n_movies, n_factors)
		self.u.weight.data.uniform_(0,0.05)
		self.m.weight.data.uniform_(0,0.05)
		
	def forward(self, cats, conts):
		users,movies = cats[:,0],cats[:,1]
		u,m = self.u(users),self.m(movies)
		return (u*m).sum(1).view(-1, 1)
```

As the name of this class would suggest we're doing a dot product of embedding matrices. 

```users,movies = cats[:,0],cats[:,1]``` gives us a minibatch of users and movies. We only look at categorical variables for embeddings. `conts` refers to continous variables. 

This minibatch size will be determined by the batchsize that you set. According to [this](https://arxiv.org/abs/1609.04836) paper a large batch size can actually the quality of the model. But according to [this](https://arxiv.org/abs/1706.02677) paper a large batch size assists model training. There is no consensus at the moment. Many people are reporting [contradictory results](https://stats.stackexchange.com/questions/436878/choosing-optimal-batch-size-contradicting-results). So I'm just going to go with a batch size of `64`. 

From that minibatch we want to do an array lookup in our embedding matrix. 

`self.u(users),self.m(movies)` allows us to do that array lookup. This lookupis less computionally intensive that a matrix mulitply of a one-hot encoded matrix and a weight matrix. 

`(u*m).sum(1).view(-1, 1)` is a cross product of the embeddings for users and movies and returns a single number. This is the predicted rating for that movie. 

# Training the Net

Next we need to create a `ColumnarModelData` object

```python
from fastai.collab import *
from fastai.tabular import *

user_name = 'userId'
item_name = 'movieId'
rating_name = 'rating'
x = ratings.drop([rating_name, 'timestamp'],axis=1)
y = ratings[rating_name].astype(np.float32)
data = ColumnarModelData.from_data_frame(path, val_idxs, x, y, [user_name, item_name], bs=64)

```

Then I'll setup an optimiser. I'll use stochastic gradient descent for this.  

```python
model = EmbeddingDot(n_users, n_movies).cuda()
opt = optim.SGD(model.parameters(), 1e-1, weight_decay=1e-5, momentum=0.9)
```

`optim.SGD` implements [stochastic gradient descent](https://pytorch.org/docs/stable/optim.html#torch.optim.SGD). 
Stochastistic gradient descent is computationally less intensive thatn gradient descent. 


Then we fit for a `3` epochs. 

```python
fit(model, data, 3, opt, F.mse_loss)
```

MSE loss is simply mean square error loss. This is calculated automatically.

# Adding in Bias and Dropout

Fastai creates a neural net automatically behind the scenes. You can call a [`collab_learner`](https://docs.fast.ai/collab.html#collab_learner) which automatically creates a neural network for collaborative filtering. Fastai also has options for introducing [Bias](https://dev.fast.ai/tutorial.collab#Movie-bias) and [dropout](https://www.cs.toronto.edu/~hinton/absps/JMLRdropout.pdf) through this collab learner. 

Using fastai we can create a collab learner easily:

```python
user_name = 'userId'
item_name = 'movieId'
rating_name = 'rating'

cols = [user_name, item_name, rating_name]

data = (CollabDataBunch.from_df(ratings[cols],
								user_name=user_name,
								item_name=item_name,
								rating_name=rating_name,
								seed = 42,
								valid_pct = 0.2,
								bs=2**11)) # up to batch size

y_range = ((ratings[rating_name].min(),
			ratings[rating_name].max()+0.5))

learn = collab_learner(data, n_factors=50, y_range=y_range)
```

Bias is very useful. We need to find user bias and movie bias. User bias would account for people who give high ratings for every movie. Movie bias would account for people who 
tend to give high ratings for a certain type of movie. Fastai adds in Bias automatically. 

Interestingly, fastai notes that you should be increase the `y_range` [slightly](https://youtu.be/CJKnDu2dxOE?t=2609). A [sigmoid function](https://en.wikipedia.org/wiki/Sigmoid_function) is used to ensure that the final output is between the numbers specified in `y_range`. The issue is that a sigmoid function asymtotes. So we'll need to increase our `y_range` slightly. Fastai recommends increasing by `0.5`.

![alt text](/images/pytorch_recommendation/lr_find.png)

I'm using the suggested learning rate here with a small amount of weight decay. This is the combination that I found to work really well. 

![alt text](/images/pytorch_recommendation/training.png)

We can train some more

![alt text](/images/pytorch_recommendation/more_training.png)
![alt text](/images/pytorch_recommendation/plot_losses.png)


We finally get a MSE of `0.784105`. But it's a very bumpy ride. Our loss jumps up and down considerably. That said `0.784105` is actually a better score than some of the [LibRec system](https://www.librec.net/release/v1.3/example.html) for collborative filtering. They were getting `0.91**2 = 0.83` MSE.  

It's also actually better than the model that fastai created in their [colloborative filtering lesson](https://github.com/fastai/course-v3/blob/master/nbs/dl1/lesson4-collab.ipynb). They were getting `0.814652` at their lowest. 


# Improvements

1. We can adjust the size of the embedding by sending in a dictionary called `emb_szs`. This could be a useful parameter to adjust. 

2. Content-based recommendation. Collaborative filtering is just one method of building a recommendation system. [Other methods](https://www.kaggle.com/ibtesama/getting-started-with-a-movie-recommendation-system#Content-Based-Filtering) could be more useful. A Content-based system is something I'm keeping in mind. That could look at metadata such as cast, crew, genre and director to make recommendations. I think some kind of [hybrid](https://www.kaggle.com/rounakbanik/movie-recommender-systems#Movies-Recommender-System) solution would be optimal. This would combin a content-based recommendation system and a collaborative filtering system. 

3. Collaborative filtering is thwarted by the [cold-start problem](https://en.wikipedia.org/wiki/Cold_start_(recommender_systems). To overcome this we could potentially look at the users metadata. For example we could look at things like: gender, age, city, time they accessed the site, etc. Just all the things they entered on the sign up form. Building a model on that data could be tricky, but if it works well it could be useful.








