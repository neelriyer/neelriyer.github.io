---
layout: post
title: Album Art Collage Wallpaper
---

![img](/images/album_art/created_wallpaper.jpg)

<!-- <div style="text-align: center"><img src="/images/album_art/original_wallpaper.jpg" width="500" /></div> -->

I'm often asked: *"What music do you listen to?"*. And I'd like to say something cool like 'The Clash' or 'Black Sabbath'. But in reality I listen to a lot of uncool bands (Tears for fears for example).

To answer that question honestly, I'll need to look at my most played songs on iTunes. 

That got me thinking. Is there a way to create a wallpaper collage that consists of the top 30 or bands you actually listen to? 

Not the stuff you say you listen to. But the stuff you actually listen to. 

Well I went ahead and created it. Here's what I did. 

# Reading iTunes Data

The first step is reading the iTunes Metadata. This turns out to be suprisingly simple. 

The following code reads the iTunes Data and stores it in a pandas dataframe.

```python
ITUNES_DIR = os.path.expanduser('/Users/neeliyer/Music/iTunes/')
LIBRARY_FILE = os.path.join(ITUNES_DIR, 'Library.xml')
MUSIC_PATH = os.path.join(ITUNES_DIR, 'iTunes Media/Music/')

try:
	d = NSDictionary.dictionaryWithContentsOfFile_(LIBRARY_FILE)
except:
	with open(LIBRARY_FILE, 'rb') as f:
		d = plistlib.load(f)
tracks = d['Tracks']

full = pd.DataFrame.from_dict(tracks, orient='index')
full.head()
```

# Most Played Albums

The next step is figuring out what are the most popular albums. To do this I found the aggregate playcount for all songs in the album. Then I sorted by playcount in descending order. 

```python
most_popular = full[['Album', 'Artist', 'Name', 'Play Count', 'Location']]
most_popular['Name'] = most_popular['Name'].apply(lambda x: [x])
most_popular = most_popular.groupby(['Album', 'Artist'], as_index = False).agg({
	'Location': 'first',
	'Play Count': 'sum',
	'Name': 'sum'
	})
most_popular = most_popular.sort_values(by=['Play Count'], ascending=False)
most_popular.head()
```
It's important to groupby both Album and Artist. There are several Artists in my library released an album called "Greatest Hits". I want to treat these as separate albums.

The downside with this method is that: What if an artist releases two albums with the same title? In that case the playcounts would be aggregated across both albums. But would Kanye really reuse an old album name? I doubt it. 

# Get Album Artwork

We can use ffmpeg to [obtain the artwork](https://stackoverflow.com/questions/13592709/retrieve-album-art-using-ffmpeg). I'm calling ffmpeg through the `subprocess` module here. 

```python
for i in range(len(most_popular)):
	subprocess.call(['ffmpeg','-i',\
	  most_popular['Location'][i],\
	  'artwork/'+str(i)+'.jpg'])
```

This will obtain the album artwork for the song and store it in the folder called `artwork`.

# Create the Collage

This part was a bit daunting for me at first. Then I found a script on github that did everything for me. That script will create a collage for us from the artwork we retrieved earlier. 

I've completed forgotten the github repo that I copied the script from and I can't find it on github anymore. I've uploaded the script to my own github [repo](https://github.com/spiyer99/album_artwork/blob/master/wallpaper.py). Please note I didn't make this script. 

We can the script pretty easily.

```
python wallpaper.py 2560 1600 -d '/artwork' -s 320
```
The output is saved as `wallpaper.jpg` in the same folder as the script.

![img](/images/album_art/created_wallpaper.jpg)

<!-- <div style="text-align: center"><img src="/images/album_art/created_wallpaper.jpg" width="500" /></div> -->

It looks good! Now I just need to get this script to automatically update. But that's for another evening. 

The full code for this can be found on [Github](https://github.com/spiyer99/album_artwork)






