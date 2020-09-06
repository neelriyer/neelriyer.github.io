---
layout: post
title: Data Transformation in Pyspark
---

Pyspark requires you to think about data differently. 

Instead of looking at a dataset row-wise. Pyspark encourages you to look at it column-wise. This was a difficult transition for me at first. It took me a long time to really understand certain Data Transformations in Pyspark. 

I'll tell you the main tricks I learned so you don't have to waste your time searching for the answers. 


# Dataset

I'll be using the [Hazardous Air Pollutants](https://www.kaggle.com/epa/hazardous-air-pollutants) dataset from Kaggle.

This Dataset is `8,097,069` rows. It's *enourmous*. 

```
df = spark.read.csv('epa_hap_daily_summary.csv',inferSchema=True, header =True)
df.show()
```
![img](images/pyspark_hacks/spark_show.png)



# Conditional If Statement

The first transformation we'll do is a conditional if statement transformation. This is as follows: if a cell in our dataset contains a particular string we want to change the cell in another column.

Basically we want to go from this:

![img](images/pyspark_hacks/pyspark_conditional_if_before.png)

To this:

![img](images/pyspark_hacks/pyspark_conditional_if_after.png)

In the `local site name` contains the word `police` then we set the `is_police` column to `1`. Otherwise we set it to `0`.

This kind of condition if statement is fairly easy to do in Pandas. We would use [`pd.np.where`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.where.html) or a [`df.apply`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.apply.html). In the worst case scenario, we could even iterate through the rows. We can't do any of that in Pyspark.

In Pyspark we can use the [`F.when`](https://spark.apache.org/docs/2.1.0/api/python/pyspark.sql.html#pyspark.sql.functions.when) statement or a [`UDF`](https://spark.apache.org/docs/2.2.0/api/python/pyspark.sql.html#pyspark.sql.functions.udf). This allows us to achieve the same result as above.

```
from pyspark.sql import functions as F

df = df.withColumn('is_police', F.when(F.lower(F.col('local_site_name')).contains('police'), F.lit(1)).\
                                otherwise(F.lit(0)))
df.select('is_police', 'local_site_name').show()
```
![img](images/pyspark_hacks/pyspark_conditional_if_after.png)



Now suppose we want to extend what we've done above. This time, however, if we see any one of 3 strings then we'll change the cell in another column. 

If any one of strings: `'Police', 'Fort' , 'Lab'` are in the `local_site_name` then we'll mark that row as `High Rating`.

The [`rlike`](https://spark.apache.org/docs/2.2.0/api/python/pyspark.sql.html#pyspark.sql.Column.like) function combined with the `F.when` function we saw earlier allow to us to do just that.


```
parameter_list = ['Police', 'Fort' , 'Lab']

df = df.withColumn('rating', F.when(F.col('local_site_name').rlike('|'.join(parameter_list)), F.lit('High Rating')).otherwise(F.lit('Low Rating')))
df.select('rating', 'local_site_name').show()
```
![img](images/pyspark_hacks/pyspark_conditional_if_rlike.png)


`F.when` is actually useful for a lot of different things. In fact you can even do a chained `F.when`:


```
df = df.withColumn('rating', F.when(F.lower(F.col('local_site_name')).contains('police'), F.lit('High Rating'))\
                              .when(F.lower(F.col('local_site_name')).contains('fort'), F.lit('High Rating'))\
                              .when(F.lower(F.col('local_site_name')).contains('lab'), F.lit('High Rating'))\
                              .otherwise(F.lit('Low Rating')))


df.select('rating', 'local_site_name').show()
```

![img](images/pyspark_hacks/pyspark_conditional_if_rlike.png)


This achieves exactly the same thing we saw in the previous example. However, it's more code to write. It's more code to maintain. 

I prefer the `rlike` method discussed above. 


# Remove whitespace

Whitespace can be really annoying. It really affects string matches and can cause unnecessary bugs in queries. 

In my opinion it's a good idea to remove whitespace as soon as possible.

[`F.trim`](https://spark.apache.org/docs/2.1.0/api/python/pyspark.sql.html#pyspark.sql.functions.trim) allows us to do just that. It will remove all the whitespace for every row in the specified column.

```
df = df.withColumn('address', F.trim(F.col('address')))
df.show()
```

![img](images/pyspark_hacks/whitespace_after.png)

# Remove Null Rows for a Particular Column

Suppose we want to remove null rows on only one column. If we encounter `NaN` values in the `pollutant_standard` column drop that entire row. 

This can accomplished fairly simply.

```
filtered_data = df.filter((F.col('pollutant_standard').isNotNull())) # filter out nulls
filtered_data.count()
```

The conditional OR parameter allows to remove rows where we `event_type` or `site_num` are `NaN.`

This is [referred](https://stackoverflow.com/questions/3154132/what-is-the-difference-between-logical-and-conditional-and-or-in-c) to as `|`.


```
filtered_data = df.filter((F.col('event_type').isNotNull()) | (F.col('site_num').isNotNull())) # filter out nulls
filtered_data.count()
```


[`df.na.drop`](https://spark.apache.org/docs/2.2.0/api/python/pyspark.sql.html#pyspark.sql.DataFrame.dropna) allows us to remove rows where all our columns are `NaN`. 


```
filtered_data = df.na.drop(how = 'all') # filter out nulls
filtered_data.show()
```












