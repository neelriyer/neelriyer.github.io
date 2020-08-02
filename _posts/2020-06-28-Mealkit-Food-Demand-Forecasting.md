---
layout: post
title: Forecasting Food Demand
---

![alt text](/images/mealkit/meal_kit_intro.png)

Applying Neural Networks to the Meal kit Market.

So this is going to overfit. Time series problems usually struggle with overfitting. This entire exercise became more of a challenge to see how I could prevent overfitting in time series forecasting. 

I added [weight decay](https://www.fast.ai/2018/07/02/adam-weight-decay/) and [dropout](https://www.cs.toronto.edu/~hinton/absps/JMLRdropout.pdf). This should work to prevent overfitting. The network has embedding layers for categorical variables (which I vary in size) followed by dropout and [batch normalisation](https://arxiv.org/abs/1502.03167) (for continuous variables).

According to [this article](https://medium.com/@lankinen/fast-ai-lesson-6-notes-part-1-v3-646edf916c04) ideally, you want lower amounts of dropout and larger amounts of weight decay. 


# Dataset

The data is given by a [meal kit company](https://datahack.analyticsvidhya.com/contest/genpact-machine-learning-hackathon-1/#ProblemStatement). As food is perishable, planning and demand prediction is extremely important. 

Getting this wrong can spell disaster for a meal kit company. Replenishment is typically done on a weekly basis. We need to forecast demand for the next 10 weeks. 

<!-- # Feature Engineering

Since I’m going to be using a neural network feature engineering is not necessarily required. The features will be automatically created by the network. 

But I’m probably going to be deploying this network in a resource constrained environment (my laptop). So, I think, a bit of feature engineering will go a long way in training the network [quickly and efficiently](https://stats.stackexchange.com/questions/349155/why-do-neural-networks-need-feature-selection-engineering).

I added two new features:
- `price_diff_percent`
The percent difference between the base price and the checkout price

- `email_plus_homepage`
Tells us if the meal was promoted over email and on the homepage.
 -->

# Pre-processing

Thanks to Fastai, normalizing, filling missing values and encoding categorical variables is now a relatively simple process.

```python
# Fill Missing values
# Encode categorical variables
# Normalize continous variables
procs=[FillMissing, Categorify, Normalize]

cont_vars = [i for i in ['checkout_price', 
             'base_price', 
             'Elapsed',
             'week_sin', 
             'week_cos',  
             'price_diff_percent'] if i in train_df.columns and i in test_df.columns]

cat_vars = [i for i in ['week', 'center_id', 'meal_id',
       'emailer_for_promotion', 'homepage_featured', 
       'category', 'cuisine', 'city_code', 'region_code', 'center_type',
       'op_area', 'Year', 'Month', 'Week', 'Day', 'Dayofweek', 'Dayofyear',
       'Is_month_end', 'Is_month_start', 'Is_quarter_end', 'Is_quarter_start',
       'Is_year_end', 'Is_year_start',
       'email_plus_homepage'] if i in train_df.columns and i in test_df.columns]

dep_var = 'num_orders'
df = train_df[cat_vars + cont_vars + [dep_var,'Date']].copy()


bs = 2**11 # max this out
path = Path('.')

# create tabular data bunch
# validation set will be 5000 rows (ordered)
# label cls
data = (TabularList.from_df(df, cat_names=cat_vars, cont_names=cont_vars, procs=procs)
                .split_by_idx(list(range(1000,1000+5000))) 
                .label_from_df(cols=dep_var, label_cls=FloatList, log = True)
                .add_test(TabularList.from_df(test_df, path=path, cat_names=cat_vars, cont_names=cont_vars, procs = procs))
                .databunch(bs=bs))

```


Next we’ll create embeddings for each categorical variable. The question is though, how large should each categorical embedding be? Fastai has a good rule of thumb where the categorical embedding is given by the following:

```python
cardinality = len(df[cat_vars[0]].value_counts().index)
emb_size = min(50, cardinality//2)
```

This means, the size of the embedding would be either the number of unique values in the categorical variable, divided by 2 and rounded down. Or it could be 50. Whichever one is smallest. 

Our final embedding size dictionary looks like this

```python
emb_szs = {cat_vars[i]:min(len(df[cat_vars[i]].value_counts().index)//2, min_size) for i in range(1, len(cat_vars))}
```
```
{'Day': 15,
 'Dayofweek': 0,
 'Dayofyear': 50,
 'Is_month_end': 1,
 'Is_month_start': 1,
 'Is_quarter_end': 1,
 'Is_quarter_start': 1,
 'Is_year_end': 1,
 'Is_year_start': 1,
 'Month': 6,
 'Week': 26,
 'Year': 1,
 'category': 7,
 'center_id': 38,
 'center_type': 1,
 'city_code': 25,
 'cuisine': 2,
 'email_plus_homepage': 1,
 'emailer_for_promotion': 1,
 'homepage_featured': 1,
 'meal_id': 25,
 'op_area': 15,
 'region_code': 4}
```

We’ll need to adjust our model architecture. This is always the hardest part. Several articles highlight the importance of getting the model architecture right. In many ways this could be seen as the [‘new’ feature engineering](https://smerity.com/articles/2016/architectures_are_the_new_feature_engineering.html). 

[This paper](https://arxiv.org/abs/1803.09820) by [Leslie Smith](https://scholar.google.com/citations?user=pwh7Pw4AAAAJ&hl=en) poses an interesting way to approach to select hyper parameters in a more disciplined way. I’ll be modelling the implemention from [this kaggle kernel](https://www.kaggle.com/qitvision/a-complete-ml-pipeline-fast-ai). 

We need to find the optimal learning rate, weight decay and embedding dropout. According to Leslie Smith, to select the optimal hyper parameters we need to run a learning rate finder for a few different values of weight decay and dropout. Then we select the largest combination that has the lowest loss, highest learning rate (before rapidly increasing) and highest weight decay. 

That’s a lot to consider. What’s more there are a few other hyperparameters we haven’t considered. For those I’ll be borrowing a model architecture with relatively large layers. [This model](https://github.com/fastai/fastai/blob/master/courses/dl1/lesson3-rossman.ipynb) was used to rank 3rd place in the Rossman Kaggle competition. 

Finally, I’ll need to consider batch size. According to Leslie this should be set as high as possible to fit onto all available memory. I’m only too happy to do that. That reduces my training time significantly. 

I used the learning rate finder in fastai to visualise the loss as we change the model architecture. From there, I created a rudimentary gridsearch. I don’t want to implement a crazy in depth grid search- that would be computationally expensive. A more manual approach is best I think. 

```python
from itertools import product
from tqdm.notebook import tqdm

def get_learner(emb_szs=emb_szs, layers=[1000,500], ps=[0.02,0.04], emb_drop=0.08):

  return (tabular_learner(data,
                          layers=layers,
                          ps=ps,
                          emb_drop=emb_drop,
                          y_range=y_range,
                          emb_szs=emb_szs,
                          metrics=exp_rmspe))


lrs = []
losses = []
wds = []
ps = []
layers = []
iter_count = 600 # anything over 300 seems to work well.
curr_wd = 1e-3
layers = [1000,500]
ps = [0.002,0.02]
emb_drop = 0.04


params = {
    'wd':[i for i in np.linspace(0,0.6,7)]
}

parameter_combinations = []


for i in tqdm(list(product(*params.values()))):

  curr_wd = i[0]

  print("curr_wd = {}".format(i[0])

  learner = get_learner(emb_szs=emb_szs, layers = layers, ps = ps, emb_drop = emb_drop) 

  learner.lr_find(wd=curr_wd, num_it=iter_count)

  lrs.append(learner.recorder.lrs)
  losses.append(learner.recorder.losses)

  combination = [[curr_wd]]
  parameter_combinations += combination

```

Once we plot all out different combinations of model architectures, things become a little clearer. 

![alt text][grid_search_plot]

Loss spikes earlier if we choose a model architecture with 0 weight decay. From the options a weight decay of `0.6` allows us to train a reasonably high learning rate with the lowest loss. 

While the learning rate that corresponds with the lowest level of loss is in the 1e-1 region, I won’t be using that learning rate. Instead I’ll be choosing the 1e-2 value for the learning rate. That’s a value on the safe side of explosion. This has been [shown](https://sgugger.github.io/how-do-you-find-a-good-learning-rate.html ) to help in training. 

Here’s our final model:
```python
learn = get_learner(emb_szs=emb_szs, layers = [1000,500], ps = [0.2,0.4], emb_drop = 0.04)
```
```
TabularModel(
  (embeds): ModuleList(
    (0): Embedding(146, 26)
    (1): Embedding(78, 38)
    (2): Embedding(52, 25)
    (3): Embedding(3, 1)
    (4): Embedding(3, 1)
    (5): Embedding(15, 7)
    (6): Embedding(5, 2)
    (7): Embedding(52, 25)
    (8): Embedding(9, 4)
    (9): Embedding(4, 1)
    (10): Embedding(31, 15)
    (11): Embedding(4, 1)
    (12): Embedding(13, 6)
    (13): Embedding(53, 26)
    (14): Embedding(32, 15)
    (15): Embedding(2, 0)
    (16): Embedding(146, 50)
    (17): Embedding(3, 1)
    (18): Embedding(3, 1)
    (19): Embedding(3, 1)
    (20): Embedding(3, 1)
    (21): Embedding(3, 1)
    (22): Embedding(3, 1)
    (23): Embedding(4, 1)
  )
  (emb_drop): Dropout(p=0.04, inplace=False)
  (bn_cont): BatchNorm1d(6, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (layers): Sequential(
    (0): Linear(in_features=256, out_features=1000, bias=True)
    (1): ReLU(inplace=True)
    (2): BatchNorm1d(1000, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (3): Dropout(p=0.2, inplace=False)
    (4): Linear(in_features=1000, out_features=500, bias=True)
    (5): ReLU(inplace=True)
    (6): BatchNorm1d(500, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (7): Dropout(p=0.4, inplace=False)
    (8): Linear(in_features=500, out_features=1, bias=True)
  )
)
```


[grid_search_plot]: /images/mealkit/grid_search_plot.png "grid_search_plot"

[comparison_traditional_methods_plot]: /images/mealkit/comparison_traditional_methods_plot.png "comparison_traditional_methods_plot"

[5_cycles]: /images/mealkit/5_cycles.png "5_cycles"

[plot_loss]: /images/mealkit/plot_loss.png "plot_loss"

[second_loss]:/images/mealkit/second_loss.png "second_loss"

[10_cycles]:/images/mealkit/10_cycles.png "10_cycles"

I’ll be using learning rate annealing here. That’s [shown to work well](https://sgugger.github.io/the-1cycle-policy.html)  

![alt text][5_cycles]

![alt text][plot_loss]

I’ll keep fitting more cycles until validation starts to increase.I'll save the model after fitting for a few epochs. That way I get use the best model later for inference. 

![alt text][10_cycles]
![alt text][second_loss]

The best I was able to get was a validation loss of about 0.29 (rounding up).

In a similar fashion to Martin Alacron's [article](https://www.martinalarcon.org/2018-12-31-b-water-pumps/
) I'd like to compare the performance of the neural network to more traditional approaches.


# Other Approaches

XGBoost, Random Forest Regressor and LightGBM. How do they perform relative to a neural network?

I will be using more or less the same data that the neural network used. Fastai has excellent pre-processing methods already built in. 

However, Fastai's categorical encoding is slightly odd. Fastai creates a dictionary from the categorical values to their encoding values. At inference time the categorical values [are swapped](https://forums.fast.ai/t/fastai-v2-code-walk-thru-8/55068) for the encoding values. 

This is very smart and very useful. But it makes it slightly difficult to use Fastai pre-processed data with models outside of the Fastai ecosystem. 

To fix this, I created a simple script to convert the Fastai Tabular Data Bunch to data that we can feed to another model. 

```python
# inspired by https://www.martinalarcon.org/2018-12-31-b-water-pumps/
class convert_tabular_learner_to_df():

  def __init__(self, cat_names, tabular_data_bunch):
    self.cat_names = cat_names
    self.tabular_data_bunch = tabular_data_bunch

  def driver(self):

    # convert tabular data to dataframe
    X_train, y_train = self.list_to_df(self.tabular_data_bunch.train_ds)
    X_valid, y_valid = self.list_to_df(self.tabular_data_bunch.valid_ds)

    # label encode data
    encoder = BinaryEncoder(cols = self.cat_names)
    X_train = encoder.fit_transform(X_train)
    X_valid = encoder.transform(X_valid)

    return X_train, X_valid, y_train, y_valid

  def list_to_df(self, tabular_learner):

    # create X df
    x_vals = np.concatenate([tabular_learner.x.codes, tabular_learner.x.conts], axis=1)
    cols = tabular_learner.x.cat_names + tabular_learner.x.cont_names
    x_df = pd.DataFrame(data=x_vals, columns=cols)

    # reorder cols
    x_df = x_df[[c for c in tabular_learner.inner_df.columns if c in cols]]

    # create y labels
    cols = [i.obj for i in tabular_learner.y]
    y_vals = np.array(cols, dtype="float64")

    return x_df, y_vals


```

Now we’ll throw a bunch of regressors at the data. Each using the default values. I’d like to know how the neural network performs relative to the standard approach of regression. Is it better? Worse?

```python
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import lightgbm as lgb

def rmspe_calc(y_true, y_pred):
    # Compute Root Mean Square Percentage Error between two arrays.
    return np.sqrt(np.mean(np.square(((y_true - y_pred) / y_true)), axis=0))


models = [
  xgb.XGBRegressor(),
  lgb.LGBMRegressor(),
  RandomForestRegressor()
]

results = pd.DataFrame(columns=["Regressor", "RMSPE"])

for model in models:

  name = model.__class__.__name__
  
  model.fit(X_train, y_train)

  rmspe = rmspe_calc(y_valid, model.predict(X_valid))
  
  df2 = pd.DataFrame(
    {"Regressor": name, \
     "RMSPE": rmspe*100}, index = [0]
  )

  results = results.append(df2, ignore_index = True)
```

Now for the results.

![alt text][comparison_traditional_methods_plot]


On this occasion it seems that the other models outperformed the neural network. Despite this, deep learning with category embeddings are very popular on [kaggle](https://www.kaggle.com/c/rossmann-store-sales/discussion/17974). I may need to vary the amount of dropout and weight decay that I'm using. But for now RandomForestRegressor is the best model in terms of RMSPE. 


# Improvements

1. Model Architecture. I’d like to vary more hyperparamters, while avoiding a costly grid search. This could be the single most useful thing in improving the model further. 

Fastai explictly warns you to not reduce parameters to avoid overfitting. Instead use dropout and weight decay liberally. 

I’ve tried to do that here. But I still ended up overfitting slightly. Varying hyperparamters could probably assist in reducing overifitting further still. 

Specifically, I could probably benefit from varying dropout. I'd like to vary the dropout for the embedding layer and more importantly the probability of dropout.

[This paper](https://scholarworks.uark.edu/cgi/viewcontent.cgi?referer=https://www.google.com/&httpsredir=1&article=1028&context=csceuht) speaks to the effectiveness of dropout in large *deep* neural networks. Perhaps making the network deeper and applying dropout more liberally could improve the performance?

Gridsearch could be implemented randomly. This is called random search. I could do this using [skorch](https://skorch.readthedocs.io/en/stable/user/quickstart.html#grid-search) potentially.

2. I’d also like to try Prophet from Facebook. It’s an open source tool for time series forecasting. I’d like to see how that performs relative to this neural network. 

3. Blending. A first place solution on kaggle used a neural network blended with a lightGBM model. This could be promising for future research.

The full code for this is available on [Github](https://github.com/spiyer99/spiyer99.github.io/blob/master/nbs/medium_food_demand_prediction_mealkit.ipynb)



