---
layout: post
title: Segment Satellite Imagery using NDVI
---

_Use rasterio to Segment Canopy Cover from Soil Easily_

In this post we'll be trying to segment canopy cover and soil on satellite imagery. So ideally we want to go from a regular satellite image:

![alt text](/images/satellite_segmentation_ndvi/example_image_start.png)

<sub>(I've shown th RGB form for visualisation purposes. In reality I'll be using the `.tif` files)</sub>

To this:

![alt text](/images/satellite_segmentation_ndvi/example_image_end.png)

<sub>The orange is soil. The red is vegetation.</sub>

We'll be borrowing ideas from [this paper](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0196605). It implements exactly what we need for sorghum. 

I'll also be using ideas from my [previous](https://towardsdatascience.com/segment-satellite-images-using-rasterio-and-scikit-learn-fc048f465874) blog post on this topic.


# NDVI

As the paper notes we'll need to extract the Normalized difference vegetation index. This is a useful index for vegetation. Here's an example of a NDVI image:

![ndvi](https://i2.wp.com/www.geoawesomeness.com/wp-content/uploads/2016/02/NDVI-image-Drone-Remote-Sensing-Geoawesomeness.png?resize=975%2C708&ssl=1). 

And here's the formula:

<img src="https://www.researchgate.net/publication/342413913/figure/fig2/AS:905930921226240@1593002171931/Formula-used-to-calculate-the-normalized-difference-vegetation-index-NDVI.ppm" alt="img" align = "center" width="300"/> 

<!-- ![formula]() -->

Here's my code to obtain the NDVI image in a numpy array.

```python
def get_ndvi(red_file):

    nir_file = os.path.dirname(red_file) + '/nir.tif'
    band_red = rasterio.open(red_file)
    band_nir = rasterio.open(nir_sfile)
    red = band_red.read(1).astype('float64')
    nir = band_nir.read(1).astype('float64')

    # ndvi calculation, empty cells or nodata cells are reported as 0
    ndvi=np.where( (nir==0.) | (red ==0.), -255 , np.where((nir+red)==0., 0, (nir-red)/(nir+red)))

    return ndvi

```

The NDVI is calculated for a certain region of interest. This is defined in the paper. 

> 
For each field plot, a region of interest (ROI) was established manually by choosing the central two rows and mean value of vegetation indices were extracted corresponding to each plot.
>

Here's my code to do exactly that.

```python

# get average ndvi of center two rows
  def get_region_of_interest(ndvi, multiplier = 1/2):

    # undo the background adjustment
    region = ndvi.copy()
    region = np.where(region==-255, 0, region)

    # mean of center rows
    center_row1 = np.mean(region[int((multiplier)*len(region))])
    center_row2 = np.mean(region[int((multiplier)*len(region))+1])

    # mean of both rows
    mean = (center_row1.copy()+center_row2.copy())/2

    return mean

```

# Fractional Vegetation Cover

The paper also makes reference to fractional vegetation cover. I'll calculate this by setting a threshold. Any NDVI value above that threshold will be vegetation. Anything below that will be soil. 

Here's the code:

```python
THRESHOLD = 0.3

def get_fc(self, ndvi):

    ndvi_copy = ndvi.copy()

    vegetation = np.where(ndvi_copy > THRESHOLD, 1, 0)
    vegetation_count = np.count_nonzero(vegetation)

    total = ndvi_copy.shape[0]*ndvi_copy.shape[1]
    fractional_cover = vegetation_count/total

    return fractional_cover
```

We'll need to change the threshold value later on.


# Plot Fractional Vegetation Cover vs NDVI

Now we can recreate the plots on page 7 of the paper. We'll be plotting fractional vegeation cover vs NDVI for each image. 

We also want to plot a line of best fit calculated by least squares. Then we extract the R^2 value associated with that regression.

This turned out to be slightly complex. We're dealing with many different numpy arrays so that is to be expected I suppose. This function is part of a class. See the [full code](https://github.com/spiyer99/spiyer99.github.io/blob/master/nbs/blog_post_segment_satellite_ndvi.ipynb) for details. 

```python
def plot_fc_vs_ndvi(self, fc, ndvi):

    y = np.array(fc).reshape(1, -1)
    x = np.array(ndvi).reshape(1,-1)

    slope, intercept, r_value, p_value, std_err = stats.linregress(x,y)

    # append slopes and intercepts to global variables
    self.slopes += [slope]
    self.intercepts += [intercept]

    x = np.linspace(min(ndvi),max(ndvi),100)
    f, ax = plt.subplots(1,1,figsize=(10,10))
    ax.plot(x, slope*x+intercept, '-r', label='fc='+str(round(slope, 2))+'*ndvi+'+str(round(intercept, 2)), color='black')
    ax.set_title('Fractional Cover vs NDVI at threshold of '+ str(self.ndvi_threshold))

    scatter = ax.scatter(x=ndvi, y=fc, c=self.roi_ndvi_pixel_count, edgecolors='black', s=[100 for i in range(len(ndvi))])
    legend = ax.legend(*scatter.legend_elements(), loc="upper left", title="Region of Interest Pixel count")
    ax.add_artist(legend)

    ax.set_xlabel('Normalized difference vegetation index (NDVI)')
    ax.set_ylabel('Fractional Cover (fc)')
    # ax.text(0.6, 0.5,s='R^2 = {}'.format(round((r_value**2), 4)), fontdict={'fontsize':14, 'fontweight':'bold'})
    ax.text(min(ndvi)+0.8*(max(ndvi)-min(ndvi)), min(fc)+0.2*(max(fc)-min(fc)),s='R^2 = {}'.format(round((r_value**2), 4)), fontdict={'fontsize':14, 'fontweight':'bold'})
    f.savefig('fc_vs_ndvi_plots/fc_vs_ndvi_'+str(self.plot_counter)+'.jpg')
    self.plot_counter +=1
    f.show()
```

Now we can change the `threshold` value set earlier and see how that affects the regression. The paper notes that we should select the `threshold` value which has the best regression model.

We'll run this code for all images and plot the results.

<img src="/images/satellite_segmentation_ndvi/fractioncal_cover_vs_ndvi_matrix_colour_coded_red2.png" alt="img" width="750"/> 

<!-- ![alt text](/images/satellite_segmentation_ndvi/fractioncal_cover_vs_ndvi_matrix_colour_coded_red2.png) -->

From this we can see that the highest R^2 is associated with a threshold of 0.45. 

We can apply this threshold to all images. Any NDVI value greater than 0.45 is vegetation. Anything below 0.45 is soil. 


# Create Binary Array

Using the 0.45 threshold we can create a binary array. This is the second image shown in the introduction. 


This bit of code does exactly that. The threshold is defined by in the `__init__` function. See the [full code](https://github.com/spiyer99/spiyer99.github.io/blob/master/nbs/blog_post_segment_satellite_ndvi.ipynb) for details. 

```python
def create_mask(self, red_file):

    nir_file = os.path.dirname(red_file) + '/nir.tif'
    band_red = rasterio.open(red_file)
    band_nir = rasterio.open(nir_file)
    red = band_red.read(1).astype('float64')
    nir = band_nir.read(1).astype('float64')

    # get raw ndvi and save as jpg
    self.raw_ndvi = np.where((nir+red)==0., np.nan, (nir-red)/(nir+red))
    
    # create canopy cover mask
    self.canopy_cover = np.where(np.isnan(self.raw_ndvi), np.nan, np.where(self.raw_ndvi<self.ndvi_threshold, 0, 1))
    self.canopy_cover = np.ma.masked_where(np.isnan(self.canopy_cover), self.canopy_cover)

    # show ndvi mask and save it as jpg
    print('canopy cover')
    print(np.unique(self.canopy_cover))

    return self.canopy_cover

```

Here's the output in a cleaner form. The first row is the thresholded canopy cover. The second row is the RBG satellite image.

![alt text](/images/satellite_segmentation_ndvi/9bbd67cfb0d660551d74decf916b2df2_ndvi_thresholded_0-45.png)


![alt text](/images/satellite_segmentation_ndvi/ea36717ca661ca3cca59d5ea43a81afc_ndvi_thresholded_0-45.png)

We can see that it does a pretty decent job of separating canopy cover from soil. This is much better than my [previous attempt](https://spiyer99.github.io/Kmeans-Clustering-Satellite-Imagery/) with K-Means Clustering.


# Conclusion

In this blog post, I described a way you can segment satellite imagery using NDVI. 

I did this work for a [small startup in Sydney](https://flurosat.com/). I could not have done this without their help. I learned so much from them. 

The full code is on [Github](https://github.com/spiyer99/spiyer99.github.io/blob/master/nbs/blog_post_segment_satellite_ndvi.ipynb).

I hope this helps people out there. Please reach out to me on [twitter](https://twitter.com/neeliyer11) if I've made a mistake somewhere. Thanks!



