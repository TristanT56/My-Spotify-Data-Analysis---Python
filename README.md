# My Spotify Data Analysis - Python


### Table of Contents

* [Introduction](#chapter1)


* [A) Cleaning & Peparation](#chapter2)
    * [Table 1 - My Spotify historical data](#section_2_1)
    * [Table 2 - The audio features of my tracks](#section_2_2)


* [B) Exploratory Data Analysis](#chapter3)
    * [Creating a function for EDA visualisation](#section_3_0)
    * [Table 1 EDA](#section_3_1)
        * [Histograms and boxplots - Distribution of the number of plays by Artists and Tracks](#section_3_1_1)
    * [Table 2 EDA](#section_3_2)
        * [Histograms and boxplots - Part 1 - Defining features and looking for outliers that shouldn't be there](#section_3_2_1)
        * [Correlation matrix of the audio features](#section_3_2_2)
        * [Histograms and Boxplots - Part 2 - Understanding my audio features preferences](#section_3_2_3)


* [C) Analysis Part 1 - Tops](#chapter4)
    * [Top artists](#section_4_1)
    * [Top tracks](#section_4_2)
    * [Top tracks of my top 1 artist](#section_4_3)
    * [Creating a wordcloud visual with my top 100 artists](#section_4_4)
    * [My music consumption on spotify per month during the analysis period (2020-07/2021-07)](#section_4_5)
    * [Which day I listen to spotify the most?](#section_4_6)
    * [Heatmap: When do I listen to Spotify the most during the week (by days and hours)?](#section_4_7)


* [D) Analysis Part 2 - Audio features](#chapter5)
    * [The tracks corresponding to the max and min for each audio feature](#section_5_1)
    * [The audio features corresponding to all the tracks I listened to](#section_5_2)
    * [Audio features: all tracks VS top tracks](#section_5_3)
    * [The audio features corresponding to the tracks I listen to in the morning, afternoon, evening and night/party](#section_5_4)
    * [Using these findings and the audio features patterns, let's find a track I might like to listen to in the morning/afternoon/evening/and during a party at night](#section_5_5)


* [Conclusion](#chapter6)


## Introduction <a class="anchor" id="chapter1"></a>

The main goal of this project is to practice my Python skills in data science and to introduce you to another part of myself which is my music tastes.

In this project, I will analyse my spotify data from July 2020 to July 2021. 

First, I will clean and prepare the data. Then, I will do an exploratory data analysis (EDA) to better understand my data and finish cleaning some weird/wrong outliers. And finally, the analysis. I will separate the analysis into two parts: 

Part 1 - The objectives are :

   - To know which are my top artists and my top tracks.
   - To see my music consumption on spotify per month during the analysis period (2020-07/2021-07).
   - To see when I listen to the most spotify during the week (by day of the week and by hour).

Part 2 - The objectives are:
   
   - Find the tracks corresponding to the max and min for each audio feature.
   - Find the audio features corresponding to all the songs I listened to. 
   - Compare it to the audio features of my top tracks.
   - Analyse the audio features corresponding to the tracks I listen to in the morning, afternoon, evening and night/party.
   - Using these findings and only the audio features, find a track I might like to listen to in the morning/afternoon/evening/and during a party at night.


About the data:

There are two datasets, the first is my historical Spotify data (in JSON format). I got it by requesting it from my Spotify account.
The second is the audio features of the tracks I listened to (in CSV format). I got them using the Spotify API (we will see that sometimes the API couldn't find information for some tracks).

## C) Cleaning & Preparation<a class="anchor" id="chapter2"></a>

### Table 1 - My Spotify historical data: <a class="anchor" id="section_2_1"></a>


```python
#Let's import my spotify historical data into a DataFrame and explore with .head() and .info()
import pandas as pd
spotify_tt = pd.read_json(r'C:\Users\Tristan\Documents\DATA\spotify_project\StreamingHistory0.json', orient = 'records')
spotify_tt.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>endTime</th>
      <th>artistName</th>
      <th>trackName</th>
      <th>msPlayed</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2020-07-16 15:43</td>
      <td>Rich Mullins</td>
      <td>Hold Me Jesus</td>
      <td>8631</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2020-07-16 15:43</td>
      <td>Paolo Conte</td>
      <td>L'Orchestrina</td>
      <td>198840</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2020-07-16 15:53</td>
      <td>Josh Wilson</td>
      <td>Savior, Please</td>
      <td>8540</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2020-07-17 09:29</td>
      <td>Isaac Delusion</td>
      <td>fancy</td>
      <td>31861</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2020-07-17 09:30</td>
      <td>B77</td>
      <td>Fleur</td>
      <td>23520</td>
    </tr>
  </tbody>
</table>
</div>




```python
spotify_tt.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 6859 entries, 0 to 6858
    Data columns (total 4 columns):
     #   Column      Non-Null Count  Dtype 
    ---  ------      --------------  ----- 
     0   endTime     6859 non-null   object
     1   artistName  6859 non-null   object
     2   trackName   6859 non-null   object
     3   msPlayed    6859 non-null   int64 
    dtypes: int64(1), object(3)
    memory usage: 214.5+ KB
    

Let's see if we can remove some unnecessary records in our table to reduce the work and time of asking the spotify api about the audio features of each track later.




```python
#Converting msPlayed to min_played and changing the column name
spotify_tt.msPlayed = spotify_tt.msPlayed /  60000
spotify_tt.rename(columns = {'msPlayed':'min_played'}, inplace=True)

#Let's see the first values:
print(spotify_tt.min_played.value_counts().sort_index().head(10))
```

    0.000000    164
    0.000017      3
    0.000067      1
    0.000083      2
    0.000100      1
    0.000133      3
    0.000183      2
    0.000200      1
    0.000233      1
    0.000300      1
    Name: min_played, dtype: int64
    


```python
#These results are strange, let's analise all that:
from matplotlib import pyplot as plt
import seaborn as sns
fig, ax = plt.subplots()
sns.histplot(spotify_tt['min_played'], ax=ax)

ax2 = plt.axes([0.4, 0.3, 0.45, 0.5], facecolor='y')
sns.histplot(spotify_tt['min_played'],binwidth=0.5, ax=ax2)
ax2.set_title('zoom : tracks < 5min')
ax2.set_xlabel('min_played (binwidth = 0.5 = 30s)')
ax2.set_ylabel('Nb of tracks')
ax2.set_xlim([0,5])
ax.set_title("Nb of tracks by duration (min played)")
ax.set_ylabel('Nb of tracks')
```






![durationtrack_png]( https://github.com/TristanT56/My-Spotify-Data-Analysis---Python/blob/main/Images%20for%20Readme%20markdown/output_8_1.png)
    



```python
spotify_tt[spotify_tt['min_played'] < 0.5].count()
```




    endTime       1960
    artistName    1960
    trackName     1960
    min_played    1960
    dtype: int64



There are 1960 tracks that were played for less than 30 seconds (0.5 min). 
This is probably when I change tracks directly after the first few seconds. 
We will drop them.


```python
#let's drop those records and validate that with assert and .shape before and after the drop
print(spotify_tt.shape)

spotify_tt = spotify_tt[spotify_tt.min_played >= 0.5]
assert spotify_tt[spotify_tt['min_played'] < 0.5].empty

print(spotify_tt.shape)
```

    (6859, 4)
    (4899, 4)
    

Ok, now let's look at the tracks with more than 8 and 10 minutes played:


```python
print(spotify_tt[spotify_tt['min_played'] > 8].count())
spotify_tt[spotify_tt['min_played'] > 8].head()
```

    endTime       49
    artistName    49
    trackName     49
    min_played    49
    dtype: int64
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>endTime</th>
      <th>artistName</th>
      <th>trackName</th>
      <th>min_played</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>39</th>
      <td>2020-08-12 07:56</td>
      <td>B.B. King</td>
      <td>Why I Sing The Blues</td>
      <td>8.623333</td>
    </tr>
    <tr>
      <th>42</th>
      <td>2020-08-12 08:10</td>
      <td>The Districts</td>
      <td>Young Blood</td>
      <td>8.680217</td>
    </tr>
    <tr>
      <th>83</th>
      <td>2020-08-12 10:48</td>
      <td>Peter Cat Recording Co.</td>
      <td>Memory Box</td>
      <td>8.064000</td>
    </tr>
    <tr>
      <th>136</th>
      <td>2020-08-13 10:52</td>
      <td>Joakim</td>
      <td>Nothing Gold - Todd Terje Remix</td>
      <td>9.024467</td>
    </tr>
    <tr>
      <th>150</th>
      <td>2020-08-13 11:41</td>
      <td>Donna Summer</td>
      <td>I Feel Love - 12" Version</td>
      <td>8.253100</td>
    </tr>
  </tbody>
</table>
</div>




```python
print(spotify_tt[spotify_tt['min_played'] > 10].count())
spotify_tt[spotify_tt['min_played'] > 10]
```

    endTime       8
    artistName    8
    trackName     8
    min_played    8
    dtype: int64
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>endTime</th>
      <th>artistName</th>
      <th>trackName</th>
      <th>min_played</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>164</th>
      <td>2020-08-14 15:17</td>
      <td>Michael Kiwanuka</td>
      <td>Cold Little Heart</td>
      <td>10.067467</td>
    </tr>
    <tr>
      <th>802</th>
      <td>2020-08-25 13:57</td>
      <td>Love De-Luxe</td>
      <td>Here Comes That Sound Again</td>
      <td>11.173433</td>
    </tr>
    <tr>
      <th>1796</th>
      <td>2020-11-10 15:42</td>
      <td>Lil Dicky</td>
      <td>Truman</td>
      <td>10.240067</td>
    </tr>
    <tr>
      <th>4169</th>
      <td>2021-04-15 21:24</td>
      <td>Le Joboscope</td>
      <td>Data scientist</td>
      <td>30.725217</td>
    </tr>
    <tr>
      <th>4170</th>
      <td>2021-04-15 21:54</td>
      <td>Le Joboscope</td>
      <td>Data analyst</td>
      <td>17.910500</td>
    </tr>
    <tr>
      <th>6211</th>
      <td>2021-07-05 15:10</td>
      <td>A suivre</td>
      <td>Beatmakers S1 (2/10) : Etienne de Crécy</td>
      <td>28.405400</td>
    </tr>
    <tr>
      <th>6213</th>
      <td>2021-07-05 15:53</td>
      <td>A suivre</td>
      <td>Beatmakers S1 (8/10) : Synapson</td>
      <td>23.405700</td>
    </tr>
    <tr>
      <th>6215</th>
      <td>2021-07-06 07:46</td>
      <td>A suivre</td>
      <td>Beatmakers S1 (2/10) : Etienne de Crécy</td>
      <td>10.871333</td>
    </tr>
  </tbody>
</table>
</div>



All seems to be normal.
For tracks > 8 minutes: they are tracks that are more than 8 minutes long (I checked on spotify). 
For tracks > 10 minutes: these are often podcasts. Or tracks that are around 8, 9 or 10 minutes long that I played back directly before the true end of the tracks.


```python
#Now, let's rename the columns and reorder the table
spotify_tt.rename(columns = {'endTime':'datetime', 'artistName': 'artist', 'trackName':'track'}, inplace= True)
spotify_tt = spotify_tt[['datetime', 'track', 'artist', 'min_played']]
spotify_tt.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>datetime</th>
      <th>track</th>
      <th>artist</th>
      <th>min_played</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>2020-07-16 15:43</td>
      <td>L'Orchestrina</td>
      <td>Paolo Conte</td>
      <td>3.314000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2020-07-17 09:29</td>
      <td>fancy</td>
      <td>Isaac Delusion</td>
      <td>0.531017</td>
    </tr>
    <tr>
      <th>7</th>
      <td>2020-07-17 09:31</td>
      <td>Plein de bisous</td>
      <td>Lewis OfMan</td>
      <td>1.056267</td>
    </tr>
    <tr>
      <th>8</th>
      <td>2020-07-17 09:32</td>
      <td>Le métro et le bus</td>
      <td>Lewis OfMan</td>
      <td>0.925917</td>
    </tr>
    <tr>
      <th>10</th>
      <td>2020-07-17 09:36</td>
      <td>La légende urbaine</td>
      <td>Voyou</td>
      <td>3.779767</td>
    </tr>
  </tbody>
</table>
</div>




```python
spotify_tt.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 4899 entries, 1 to 6858
    Data columns (total 4 columns):
     #   Column      Non-Null Count  Dtype  
    ---  ------      --------------  -----  
     0   datetime    4899 non-null   object 
     1   track       4899 non-null   object 
     2   artist      4899 non-null   object 
     3   min_played  4899 non-null   float64
    dtypes: float64(1), object(3)
    memory usage: 191.4+ KB
    


```python
#We need to convert datetime to a datetime format (we will set that as index later)
spotify_tt.datetime = pd.to_datetime(spotify_tt.datetime)
spotify_tt.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 4899 entries, 1 to 6858
    Data columns (total 4 columns):
     #   Column      Non-Null Count  Dtype         
    ---  ------      --------------  -----         
     0   datetime    4899 non-null   datetime64[ns]
     1   track       4899 non-null   object        
     2   artist      4899 non-null   object        
     3   min_played  4899 non-null   float64       
    dtypes: datetime64[ns](1), float64(1), object(2)
    memory usage: 191.4+ KB
    


```python
#Last check for missing data
spotify_tt.isna().sum()
```




    datetime      0
    track         0
    artist        0
    min_played    0
    dtype: int64




```python
#Ok it's clean now, let's save it into a new csv file:
spotify_tt.to_csv(r'C:\Users\Tristan\Documents\DATA\spotify_project\spotify_tt_clean.csv', index=False)
```

Ok, we cleaned up our historical data table and deleted 1960 unnecessary records to get our audio features with the spotify API. For this, I was helped by Vlad Gheorghe's article.
Article: https://towardsdatascience.com/get-your-spotify-streaming-history-with-python-d5a208bbcbd3

I now have a csv file with my audio features and other information, let's open it:

### Table 2 - The audio features of my tracks: <a class="anchor" id="section_2_2"></a>


```python
#Let's import the csv file and explore it
features = pd.read_csv(r'C:\Users\Tristan\Documents\DATA\spotify_project\features.csv')
features.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Unnamed: 0</th>
      <th>danceability</th>
      <th>energy</th>
      <th>key</th>
      <th>loudness</th>
      <th>mode</th>
      <th>speechiness</th>
      <th>acousticness</th>
      <th>instrumentalness</th>
      <th>liveness</th>
      <th>...</th>
      <th>tempo</th>
      <th>type</th>
      <th>id</th>
      <th>uri</th>
      <th>track_href</th>
      <th>analysis_url</th>
      <th>duration_ms</th>
      <th>time_signature</th>
      <th>albumName</th>
      <th>albumID</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Junk___Étienne de Crécy</td>
      <td>0.797</td>
      <td>0.579</td>
      <td>2.0</td>
      <td>-9.340</td>
      <td>1.0</td>
      <td>0.0497</td>
      <td>0.00522</td>
      <td>0.871</td>
      <td>0.0502</td>
      <td>...</td>
      <td>124.988</td>
      <td>audio_features</td>
      <td>0R1l25gRCjthLiYRraCVZW</td>
      <td>spotify:track:0R1l25gRCjthLiYRraCVZW</td>
      <td>https://api.spotify.com/v1/tracks/0R1l25gRCjth...</td>
      <td>https://api.spotify.com/v1/audio-analysis/0R1l...</td>
      <td>297293.0</td>
      <td>4.0</td>
      <td>Commercial EP 3</td>
      <td>6a4usN5WtkByPHdcOdCD23</td>
    </tr>
    <tr>
      <th>1</th>
      <td>The Letter___The Box Tops</td>
      <td>0.638</td>
      <td>0.428</td>
      <td>9.0</td>
      <td>-12.156</td>
      <td>0.0</td>
      <td>0.0687</td>
      <td>0.25200</td>
      <td>0.000</td>
      <td>0.1320</td>
      <td>...</td>
      <td>139.434</td>
      <td>audio_features</td>
      <td>6RJK553YhstRzyKA4mug09</td>
      <td>spotify:track:6RJK553YhstRzyKA4mug09</td>
      <td>https://api.spotify.com/v1/tracks/6RJK553YhstR...</td>
      <td>https://api.spotify.com/v1/audio-analysis/6RJK...</td>
      <td>112800.0</td>
      <td>4.0</td>
      <td>The Letter/Neon Rainbow</td>
      <td>08mPxuP35Db56jUUgRvGFs</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Zoom Zoom___Polo &amp; Pan</td>
      <td>0.678</td>
      <td>0.849</td>
      <td>7.0</td>
      <td>-6.983</td>
      <td>1.0</td>
      <td>0.0367</td>
      <td>0.04590</td>
      <td>0.613</td>
      <td>0.0833</td>
      <td>...</td>
      <td>94.003</td>
      <td>audio_features</td>
      <td>1gWnuGAiTk3Q4yrIbwymUK</td>
      <td>spotify:track:1gWnuGAiTk3Q4yrIbwymUK</td>
      <td>https://api.spotify.com/v1/tracks/1gWnuGAiTk3Q...</td>
      <td>https://api.spotify.com/v1/audio-analysis/1gWn...</td>
      <td>209800.0</td>
      <td>4.0</td>
      <td>Caravelle</td>
      <td>0SuFqlCe5i30Fr75ZlPQVT</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Californie___Caballero &amp; JeanJass</td>
      <td>0.920</td>
      <td>0.516</td>
      <td>11.0</td>
      <td>-8.257</td>
      <td>0.0</td>
      <td>0.1190</td>
      <td>0.38000</td>
      <td>0.000</td>
      <td>0.0698</td>
      <td>...</td>
      <td>112.013</td>
      <td>audio_features</td>
      <td>0sJX7GTLCNowidzM9HfaH5</td>
      <td>spotify:track:0sJX7GTLCNowidzM9HfaH5</td>
      <td>https://api.spotify.com/v1/tracks/0sJX7GTLCNow...</td>
      <td>https://api.spotify.com/v1/audio-analysis/0sJX...</td>
      <td>258891.0</td>
      <td>4.0</td>
      <td>Double hélice 3</td>
      <td>6nMcxKyjXxxA0WeIpOpnuJ</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Una Rosa Blanca___Ibrahim Maalouf</td>
      <td>0.463</td>
      <td>0.672</td>
      <td>8.0</td>
      <td>-6.071</td>
      <td>0.0</td>
      <td>0.0498</td>
      <td>0.72800</td>
      <td>0.743</td>
      <td>0.0993</td>
      <td>...</td>
      <td>74.916</td>
      <td>audio_features</td>
      <td>4MOCTiC5mMrJuhLFSNjiIM</td>
      <td>spotify:track:4MOCTiC5mMrJuhLFSNjiIM</td>
      <td>https://api.spotify.com/v1/tracks/4MOCTiC5mMrJ...</td>
      <td>https://api.spotify.com/v1/audio-analysis/4MOC...</td>
      <td>338000.0</td>
      <td>4.0</td>
      <td>S3NS</td>
      <td>1XWCws077Z4B9SwwUzOAfo</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 21 columns</p>
</div>




```python
features.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 1951 entries, 0 to 1950
    Data columns (total 21 columns):
     #   Column            Non-Null Count  Dtype  
    ---  ------            --------------  -----  
     0   Unnamed: 0        1951 non-null   object 
     1   danceability      1866 non-null   float64
     2   energy            1866 non-null   float64
     3   key               1866 non-null   float64
     4   loudness          1866 non-null   float64
     5   mode              1866 non-null   float64
     6   speechiness       1866 non-null   float64
     7   acousticness      1866 non-null   float64
     8   instrumentalness  1866 non-null   float64
     9   liveness          1866 non-null   float64
     10  valence           1866 non-null   float64
     11  tempo             1866 non-null   float64
     12  type              1866 non-null   object 
     13  id                1866 non-null   object 
     14  uri               1866 non-null   object 
     15  track_href        1866 non-null   object 
     16  analysis_url      1866 non-null   object 
     17  duration_ms       1866 non-null   float64
     18  time_signature    1866 non-null   float64
     19  albumName         1866 non-null   object 
     20  albumID           1866 non-null   object 
    dtypes: float64(13), object(8)
    memory usage: 320.2+ KB
    


```python
features.type.value_counts()
```




    audio_features    1866
    Name: type, dtype: int64



Ok, we will keep only the audio features that interest us:
danceability, energy, loudness, instrumentalness, acousticness, tempo and mode.

Ps :
- Speechiness is not interesting for music but rather for podcasts. Instrumentalness is the same but for music (1 = no vocals and 0 = lots of vocals).

- Liveness, we don't care if it is live or not, we are only interested in the music itself and its characteristics.

- Type, as we can see above, all my data has the type 'audio_feature' so it is not interesting.

- Id, we can also drop this as we will use 'track' and 'artist' from the first column (which we will split) to merge with my historical data table.


```python
features['track'] = features['Unnamed: 0'].str.split('___').str[0]
features['artist'] = features['Unnamed: 0'].str.split('___').str[1]
features = features[['track', 'artist', 'danceability', 'energy', 'valence', 'loudness','instrumentalness', 'acousticness', 'tempo', 'mode']]
features.head()

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>track</th>
      <th>artist</th>
      <th>danceability</th>
      <th>energy</th>
      <th>valence</th>
      <th>loudness</th>
      <th>instrumentalness</th>
      <th>acousticness</th>
      <th>tempo</th>
      <th>mode</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Junk</td>
      <td>Étienne de Crécy</td>
      <td>0.797</td>
      <td>0.579</td>
      <td>0.0415</td>
      <td>-9.340</td>
      <td>0.871</td>
      <td>0.00522</td>
      <td>124.988</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>The Letter</td>
      <td>The Box Tops</td>
      <td>0.638</td>
      <td>0.428</td>
      <td>0.9010</td>
      <td>-12.156</td>
      <td>0.000</td>
      <td>0.25200</td>
      <td>139.434</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Zoom Zoom</td>
      <td>Polo &amp; Pan</td>
      <td>0.678</td>
      <td>0.849</td>
      <td>0.5020</td>
      <td>-6.983</td>
      <td>0.613</td>
      <td>0.04590</td>
      <td>94.003</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Californie</td>
      <td>Caballero &amp; JeanJass</td>
      <td>0.920</td>
      <td>0.516</td>
      <td>0.5340</td>
      <td>-8.257</td>
      <td>0.000</td>
      <td>0.38000</td>
      <td>112.013</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Una Rosa Blanca</td>
      <td>Ibrahim Maalouf</td>
      <td>0.463</td>
      <td>0.672</td>
      <td>0.5220</td>
      <td>-6.071</td>
      <td>0.743</td>
      <td>0.72800</td>
      <td>74.916</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
#Let's make sure we don't have duplicated rows
assert features[features.duplicated()].empty

```


```python
#Let's check missing data
features.isna().sum()
```




    track                0
    artist               0
    danceability        85
    energy              85
    valence             85
    loudness            85
    instrumentalness    85
    acousticness        85
    tempo               85
    mode                85
    dtype: int64




```python
features[features.danceability.isna()]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>track</th>
      <th>artist</th>
      <th>danceability</th>
      <th>energy</th>
      <th>valence</th>
      <th>loudness</th>
      <th>instrumentalness</th>
      <th>acousticness</th>
      <th>tempo</th>
      <th>mode</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>22</th>
      <td>Vertigo Valley</td>
      <td>French 79</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>55</th>
      <td>Do It</td>
      <td>Camp Claude</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>63</th>
      <td>Your Night</td>
      <td>Con Funk Shun</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>79</th>
      <td>When The Sun Goes Down</td>
      <td>Arctic Monkeys</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>84</th>
      <td>fancy</td>
      <td>Isaac Delusion</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1860</th>
      <td>Harlem Shuffle - Alternate Take</td>
      <td>The Foundations</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1862</th>
      <td>Heaven</td>
      <td>The Blaze</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1924</th>
      <td>Slim's Night Out</td>
      <td>PillowTalk</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1933</th>
      <td>Ti voglio</td>
      <td>Ornella Vanoni</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1936</th>
      <td>Vieille branche</td>
      <td>Biga Ranx</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>85 rows × 10 columns</p>
</div>



Those are the tracks that the API did not find, let's drop them.


```python
#let's drop those records and validate the drop
print(features.shape)
features = features.dropna(subset = ['danceability'])
print(features.isna().sum())
print(features.shape)
```

    (1951, 10)
    track               0
    artist              0
    danceability        0
    energy              0
    valence             0
    loudness            0
    instrumentalness    0
    acousticness        0
    tempo               0
    mode                0
    dtype: int64
    (1866, 10)
    


```python
#Let's check the data type
features.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 1866 entries, 0 to 1950
    Data columns (total 10 columns):
     #   Column            Non-Null Count  Dtype  
    ---  ------            --------------  -----  
     0   track             1866 non-null   object 
     1   artist            1866 non-null   object 
     2   danceability      1866 non-null   float64
     3   energy            1866 non-null   float64
     4   valence           1866 non-null   float64
     5   loudness          1866 non-null   float64
     6   instrumentalness  1866 non-null   float64
     7   acousticness      1866 non-null   float64
     8   tempo             1866 non-null   float64
     9   mode              1866 non-null   float64
    dtypes: float64(8), object(2)
    memory usage: 160.4+ KB
    


```python
#Let's convert mode to integer (1= Major mode, 0= Minor mode)
features['mode'] = features['mode'].astype('int')
features.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 1866 entries, 0 to 1950
    Data columns (total 10 columns):
     #   Column            Non-Null Count  Dtype  
    ---  ------            --------------  -----  
     0   track             1866 non-null   object 
     1   artist            1866 non-null   object 
     2   danceability      1866 non-null   float64
     3   energy            1866 non-null   float64
     4   valence           1866 non-null   float64
     5   loudness          1866 non-null   float64
     6   instrumentalness  1866 non-null   float64
     7   acousticness      1866 non-null   float64
     8   tempo             1866 non-null   float64
     9   mode              1866 non-null   int32  
    dtypes: float64(7), int32(1), object(2)
    memory usage: 153.1+ KB
    


```python
#Ok, now it is clean, let's save it into a new csv file:
features.to_csv(r'C:\Users\Tristan\Documents\DATA\spotify_project\features_clean.csv', index=False)
```

## B) Exploratory Data Analysis (EDA)<a class="anchor" id="chapter3"></a>

OK, now that our two tables are pretty much clean, let's do some exploratory data analysis that will help us understand our data but also finish cleaning up possible wrong outliers.

### 1 - Creating a function for EDA visualisation: <a class="anchor" id="section_3_0"></a>


```python
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import numpy as np

#Let's create a function to plot an histgram with a boxplot
def boxdistplot(x,l, **kwargs):
    ax = sns.distplot(x, hist_kws=dict(alpha=0.3), bins = 25, color ='g', ax=l)
    ax.set_xlim([min(x)-(max(x)/20), max(x)+(max(x)/20)])
    
    kdelimit = ax.lines[0]
    xkde = kdelimit.get_xdata()
    ykde = kdelimit.get_ydata()
    left = x.mean() - x.std()
    right = x.mean() + x.std()
    ax.vlines(x.mean(), 0, np.interp(x.mean(), xkde, ykde), color='r', label ="mean: "+ str(round(x.mean(),2)))
    ax.fill_between(xkde, 0, ykde, where=(left <= xkde) & (xkde <= right), interpolate=True, facecolor='r', alpha=0.2,
                    label="std: "+ str(round(x.std(),2)))    
    ax.vlines(x.median(), 0, np.interp(x.median(), xkde, ykde), color='b', label="median: "+ str(round(x.median(),2)) )
    ax.vlines(x.quantile(0.25), 0, np.interp(x.quantile(0.25), xkde, ykde), alpha=0, label="q1: "+ str(round(x.quantile(0.25),2)) )    
    ax.vlines(x.quantile(0.75), 0, np.interp(x.quantile(0.75), xkde, ykde), alpha=0, label="q3: "+ str(round(x.quantile(0.75),2)) )    

    ax.legend(prop={"size":9})
    
    ax2 = ax.twinx()
    sns.boxplot(x=x, ax=ax2, color = 'g')
    ax2.set(ylim=(-5, 5))
```

### 2 - Table 1 EDA: <a class="anchor" id="section_3_1"></a>

We have already started the EDA for spotify_tt in the Cleaning & Preparation part as we needed to explore the min_played column to remove some unwanted records to help the spotify API get the audio features faster.

So here we will only explore artists and tracks.

#### Histograms and boxplots - Distribution of the number of plays by Artists and Tracks: <a class="anchor" id="section_3_1_1"></a>


```python
spotify_tt.artist.value_counts().describe()
```




    count    650.000000
    mean       7.536923
    std       17.470539
    min        1.000000
    25%        1.000000
    50%        2.000000
    75%        7.750000
    max      309.000000
    Name: artist, dtype: float64




```python
spotify_tt.track.value_counts().describe()
```




    count    1543.000000
    mean        3.174984
    std         4.656332
    min         1.000000
    25%         1.000000
    50%         1.000000
    75%         3.000000
    max        45.000000
    Name: track, dtype: float64




```python
#Let's use the function
fig, ax = plt.subplots(2, 1)

boxdistplot(spotify_tt.artist.value_counts(), ax[0])
plt.title('Nb of plays by artist')

boxdistplot(spotify_tt.track.value_counts(), ax[1])
plt.title('Nb of plays by track')

plt.subplots_adjust(right=1.5, top= 1.5 , hspace=0.4)
plt.show()
```

![EDAtable1_png]( https://github.com/TristanT56/My-Spotify-Data-Analysis---Python/blob/main/Images%20for%20Readme%20markdown/output_45_0.png)
    


Interpretation: 

   Artist: we can see with .describe(), that I listened to 650 different artists. The distribution of the number of plays by artist is highly right-skewed (median = 2, std = 17,47 and mean = 7.54). That means, for the majority of the artists, I listened to them only 2 times. But there are some artists that I listened many, many, times. For example, there is one artist that I listened 309 times between July 2020 and 2021 (wow!).
    
   Track: we can see with .describe(), that I listened to 1543 different tracks. The distribution of the number of plays by track is also highly right-skewed (median = 1, std = 4.66 and mean = 3.17). That means, for the majority of the tracks, I listened to them only once. But there are few tracks that I listened many times. For example, there is one track that I listened 45 times between July 2020 and 2021.

### 3 - Table 2 EDA: <a class="anchor" id="section_3_2"></a>

#### Merging the two tables:

We will merge the two tables to weight my audio features statistics with the number of plays of each track (= each recording in my historical data table). 


```python
#Let's merge the two tables
spotify_features = spotify_tt.merge(features, on = ['track', 'artist'], how='inner')
spotify_features.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>datetime</th>
      <th>track</th>
      <th>artist</th>
      <th>min_played</th>
      <th>danceability</th>
      <th>energy</th>
      <th>valence</th>
      <th>loudness</th>
      <th>instrumentalness</th>
      <th>acousticness</th>
      <th>tempo</th>
      <th>mode</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2020-07-16 15:43:00</td>
      <td>L'Orchestrina</td>
      <td>Paolo Conte</td>
      <td>3.314000</td>
      <td>0.758</td>
      <td>0.819</td>
      <td>0.748</td>
      <td>-8.698</td>
      <td>0.000184</td>
      <td>0.578</td>
      <td>117.47</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2020-08-15 19:14:00</td>
      <td>L'Orchestrina</td>
      <td>Paolo Conte</td>
      <td>3.311583</td>
      <td>0.758</td>
      <td>0.819</td>
      <td>0.748</td>
      <td>-8.698</td>
      <td>0.000184</td>
      <td>0.578</td>
      <td>117.47</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2020-08-17 13:01:00</td>
      <td>L'Orchestrina</td>
      <td>Paolo Conte</td>
      <td>3.311533</td>
      <td>0.758</td>
      <td>0.819</td>
      <td>0.748</td>
      <td>-8.698</td>
      <td>0.000184</td>
      <td>0.578</td>
      <td>117.47</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2020-08-23 15:41:00</td>
      <td>L'Orchestrina</td>
      <td>Paolo Conte</td>
      <td>3.311450</td>
      <td>0.758</td>
      <td>0.819</td>
      <td>0.748</td>
      <td>-8.698</td>
      <td>0.000184</td>
      <td>0.578</td>
      <td>117.47</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2020-08-25 14:27:00</td>
      <td>L'Orchestrina</td>
      <td>Paolo Conte</td>
      <td>3.314000</td>
      <td>0.758</td>
      <td>0.819</td>
      <td>0.748</td>
      <td>-8.698</td>
      <td>0.000184</td>
      <td>0.578</td>
      <td>117.47</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



#### Histograms and boxplots - Part 1 - Defining features and looking for outliers that shouldn't be there: <a class="anchor" id="section_3_2_1"></a>


```python
#Let's use the function
fig, ax = plt.subplots(2, 3)

boxdistplot(spotify_features['danceability'], ax[0, 0])
boxdistplot(spotify_features['energy'], ax[0, 1])
boxdistplot(spotify_features['valence'], ax[0, 2])
boxdistplot(spotify_features['loudness'], ax[1, 0])
boxdistplot(spotify_features['instrumentalness'], ax[1, 1])
boxdistplot(spotify_features['acousticness'], ax[1, 2])

plt.title('Audio features distributions')
plt.subplots_adjust(right=2, top= 1.5 , wspace=0.3, hspace=0.3)
plt.show()
```


![EDA1table21_png]( https://github.com/TristanT56/My-Spotify-Data-Analysis---Python/blob/main/Images%20for%20Readme%20markdown/output_52_0.png)
    



```python
fig, ax = plt.subplots(1, 2, squeeze=False)

boxdistplot(spotify_features['tempo'], ax[0, 0])
sns.histplot(spotify_features, x='mode', hue='mode',  ax=ax[0, 1])
perc = spotify_features['mode'].value_counts(normalize=True).mul(100).round(1).astype(str) + '%'
ax[0, 1].legend( ['major = '+ perc.iloc[0], 'minor = '+ perc.iloc[1]], title='Mode:', loc="upper center")


plt.subplots_adjust(right=1.5, top= 0.75 , wspace=0.3, hspace=0.3)
plt.show()
```

![EDA1table22_png]( https://github.com/TristanT56/My-Spotify-Data-Analysis---Python/blob/main/Images%20for%20Readme%20markdown/output_53_0.png)
    


Tempo: we can see two weird outliers on the boxplot (tempo < 80 and tempo > 200), let's check that:


```python
spotify_features[spotify_features['tempo'] < 80].drop_duplicates(subset = ['track', 'artist']).sort_values('tempo').head(5)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>datetime</th>
      <th>track</th>
      <th>artist</th>
      <th>min_played</th>
      <th>danceability</th>
      <th>energy</th>
      <th>valence</th>
      <th>loudness</th>
      <th>instrumentalness</th>
      <th>acousticness</th>
      <th>tempo</th>
      <th>mode</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>4254</th>
      <td>2021-06-07 15:48:00</td>
      <td>Lovin' Feeling</td>
      <td>French 79</td>
      <td>3.415550</td>
      <td>0.0000</td>
      <td>0.609</td>
      <td>0.0000</td>
      <td>-8.737</td>
      <td>0.627000</td>
      <td>0.13500</td>
      <td>0.000</td>
      <td>1</td>
    </tr>
    <tr>
      <th>209</th>
      <td>2020-08-12 09:33:00</td>
      <td>Awake</td>
      <td>Electric Guest</td>
      <td>5.014433</td>
      <td>0.0993</td>
      <td>0.626</td>
      <td>0.0399</td>
      <td>-8.212</td>
      <td>0.001610</td>
      <td>0.00499</td>
      <td>49.452</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4005</th>
      <td>2021-05-06 13:33:00</td>
      <td>When I Look Up</td>
      <td>Jack Johnson</td>
      <td>0.969783</td>
      <td>0.5350</td>
      <td>0.185</td>
      <td>0.4950</td>
      <td>-17.911</td>
      <td>0.000007</td>
      <td>0.70100</td>
      <td>58.583</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2180</th>
      <td>2020-10-23 12:06:00</td>
      <td>Baby Jane</td>
      <td>Arthur Dupont</td>
      <td>2.783550</td>
      <td>0.6010</td>
      <td>0.375</td>
      <td>0.4330</td>
      <td>-9.164</td>
      <td>0.005400</td>
      <td>0.64900</td>
      <td>59.993</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2991</th>
      <td>2020-12-28 21:43:00</td>
      <td>Le chat</td>
      <td>Pow Wow</td>
      <td>2.848917</td>
      <td>0.3830</td>
      <td>0.230</td>
      <td>0.4660</td>
      <td>-12.615</td>
      <td>0.000000</td>
      <td>0.79600</td>
      <td>60.067</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
spotify_features[spotify_features['tempo'] > 200].drop_duplicates(subset = ['track', 'artist']).sort_values('tempo', ascending = False).head(5)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>datetime</th>
      <th>track</th>
      <th>artist</th>
      <th>min_played</th>
      <th>danceability</th>
      <th>energy</th>
      <th>valence</th>
      <th>loudness</th>
      <th>instrumentalness</th>
      <th>acousticness</th>
      <th>tempo</th>
      <th>mode</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>3226</th>
      <td>2021-02-20 18:19:00</td>
      <td>La main à la pâte</td>
      <td>L'Entourloop</td>
      <td>2.458050</td>
      <td>0.601</td>
      <td>0.5260</td>
      <td>0.535</td>
      <td>-8.521</td>
      <td>0.001230</td>
      <td>0.096</td>
      <td>245.511</td>
      <td>0</td>
    </tr>
    <tr>
      <th>902</th>
      <td>2020-08-15 15:06:00</td>
      <td>Hell N Back</td>
      <td>Bakar</td>
      <td>3.557383</td>
      <td>0.584</td>
      <td>0.6840</td>
      <td>0.720</td>
      <td>-4.314</td>
      <td>0.000091</td>
      <td>0.312</td>
      <td>210.164</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1110</th>
      <td>2020-08-15 20:29:00</td>
      <td>It's Too Late</td>
      <td>Carole King</td>
      <td>3.886583</td>
      <td>0.450</td>
      <td>0.4420</td>
      <td>0.812</td>
      <td>-12.718</td>
      <td>0.005640</td>
      <td>0.493</td>
      <td>208.282</td>
      <td>0</td>
    </tr>
    <tr>
      <th>28</th>
      <td>2020-07-17 10:05:00</td>
      <td>On a marché sur la lune</td>
      <td>Voyou</td>
      <td>3.611767</td>
      <td>0.327</td>
      <td>0.8910</td>
      <td>0.162</td>
      <td>-6.908</td>
      <td>0.049600</td>
      <td>0.416</td>
      <td>204.851</td>
      <td>1</td>
    </tr>
    <tr>
      <th>50</th>
      <td>2020-08-12 07:25:00</td>
      <td>Foule sentimentale</td>
      <td>Chilly Gonzales</td>
      <td>1.783017</td>
      <td>0.312</td>
      <td>0.0948</td>
      <td>0.539</td>
      <td>-15.544</td>
      <td>0.864000</td>
      <td>0.935</td>
      <td>204.544</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



Tempo: There seems to be a lot of errors with Spotify's tempo recognition. We have some very low tempo songs that are the real tempo like "When I Look Up" from "Jack Johnson" (I checked on https://songdata.io/track/50lUuRklAxwZ8G7uFzX8td/When-I-Look-Up-by-Jack-Johnson ).

But others that are not ok. For example, the tempo of "Awake" from "Electric Guest" has been divided approximately by 2, its real tempo is 111BPM (on https://songdata.io/track/2QoC2SAaXpLrpVNCGJnrgd/Awake-by-Electric-Guest). 
In fact the tempo is often wrong by half or double due to elements in the music that make the recognition wrong. This is why we have very low tempo but also very high ones.

We will simply drop the extreme values (tempo=0 and tempo = 245) and keep in mind not to take tempo analysis too seriously.


```python
#Let's drop extrem tempo and validate the drop with assert and .shape
print(spotify_features.shape)

spotify_features = spotify_features[spotify_features['tempo'] > 0]
spotify_features = spotify_features[spotify_features['tempo'] <230]
assert spotify_features[spotify_features['tempo'] < 0].empty
assert spotify_features[spotify_features['tempo'] > 230].empty

print(spotify_features.shape)
```

    (4654, 12)
    (4652, 12)
    

Mode: tells if the music is in major mode or minor mode.
58.2% of my spotify history tracks are in major mode.

Danceability: to analyse whether a track is more or less danceable. 
0: not danceable / 1: highly danceable.

We can see one weird outlier on the boxplot, let's check if we have incorrect extreme values. 


```python
#Danceability < 0.2: Music you can't dance so much.
#(Awake and Shadows do not belong here)
spotify_features[(spotify_features['danceability'] < 0.2)].drop_duplicates(subset = ['track', 'artist']).sort_values('danceability').head(5)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>datetime</th>
      <th>track</th>
      <th>artist</th>
      <th>min_played</th>
      <th>danceability</th>
      <th>energy</th>
      <th>valence</th>
      <th>loudness</th>
      <th>instrumentalness</th>
      <th>acousticness</th>
      <th>tempo</th>
      <th>mode</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>209</th>
      <td>2020-08-12 09:33:00</td>
      <td>Awake</td>
      <td>Electric Guest</td>
      <td>5.014433</td>
      <td>0.0993</td>
      <td>0.62600</td>
      <td>0.0399</td>
      <td>-8.212</td>
      <td>0.00161</td>
      <td>0.004990</td>
      <td>49.452</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3531</th>
      <td>2021-04-09 12:25:00</td>
      <td>L'estasi dell'oro</td>
      <td>Ennio Morricone</td>
      <td>3.384433</td>
      <td>0.1360</td>
      <td>0.48400</td>
      <td>0.1020</td>
      <td>-12.945</td>
      <td>0.52300</td>
      <td>0.715000</td>
      <td>99.566</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1961</th>
      <td>2020-08-31 09:44:00</td>
      <td>Shadows</td>
      <td>Talisco</td>
      <td>1.857733</td>
      <td>0.1570</td>
      <td>0.00476</td>
      <td>0.0305</td>
      <td>-33.114</td>
      <td>0.95500</td>
      <td>0.952000</td>
      <td>131.936</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2654</th>
      <td>2020-12-12 21:08:00</td>
      <td>Between the Buttons</td>
      <td>French 79</td>
      <td>5.292000</td>
      <td>0.1600</td>
      <td>0.38600</td>
      <td>0.0348</td>
      <td>-12.019</td>
      <td>0.68000</td>
      <td>0.171000</td>
      <td>90.379</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4570</th>
      <td>2021-07-08 17:25:00</td>
      <td>I'm Alive</td>
      <td>The Hives</td>
      <td>2.000233</td>
      <td>0.1610</td>
      <td>0.90200</td>
      <td>0.1250</td>
      <td>-2.717</td>
      <td>0.00560</td>
      <td>0.000083</td>
      <td>86.841</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
#Danceability > 0.9: Music with a groove or something you can dance to.
#I generally agree with the results but I find it hard to believe that Quick Drive and Liquid sunshine are in the top 5...
#(credits from parcel or last night a dj saved my life are more suited to be in the top 5). 
#It may also depend on the type of dance...
#(Quick Drive and Liquid sunshine do not belong here)
spotify_features[(spotify_features['danceability'] > 0.9)].drop_duplicates(subset = ['track', 'artist']).sort_values('danceability', ascending= False).head(8)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>datetime</th>
      <th>track</th>
      <th>artist</th>
      <th>min_played</th>
      <th>danceability</th>
      <th>energy</th>
      <th>valence</th>
      <th>loudness</th>
      <th>instrumentalness</th>
      <th>acousticness</th>
      <th>tempo</th>
      <th>mode</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>4334</th>
      <td>2021-06-09 10:23:00</td>
      <td>Quick Drive</td>
      <td>Niko B</td>
      <td>3.040300</td>
      <td>0.980</td>
      <td>0.495</td>
      <td>0.950</td>
      <td>-4.997</td>
      <td>0.000118</td>
      <td>0.00731</td>
      <td>120.036</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2760</th>
      <td>2020-12-16 17:44:00</td>
      <td>Credits (feat. Dean Dawson)</td>
      <td>Parcels</td>
      <td>1.042600</td>
      <td>0.977</td>
      <td>0.499</td>
      <td>0.967</td>
      <td>-9.662</td>
      <td>0.000000</td>
      <td>0.16500</td>
      <td>115.015</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4621</th>
      <td>2021-07-10 23:43:00</td>
      <td>Tshegue</td>
      <td>Tshegue</td>
      <td>2.983050</td>
      <td>0.970</td>
      <td>0.526</td>
      <td>0.373</td>
      <td>-8.392</td>
      <td>0.000201</td>
      <td>0.01720</td>
      <td>123.971</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4100</th>
      <td>2021-05-22 09:51:00</td>
      <td>Last Night a D.J. Saved My Life</td>
      <td>Indeep</td>
      <td>5.659150</td>
      <td>0.968</td>
      <td>0.345</td>
      <td>0.954</td>
      <td>-14.170</td>
      <td>0.003440</td>
      <td>0.15400</td>
      <td>109.803</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1542</th>
      <td>2020-08-17 23:07:00</td>
      <td>Liquid Sunshine</td>
      <td>Biga Ranx</td>
      <td>3.224883</td>
      <td>0.964</td>
      <td>0.406</td>
      <td>0.723</td>
      <td>-8.259</td>
      <td>0.020600</td>
      <td>0.03290</td>
      <td>112.018</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1033</th>
      <td>2020-08-15 19:04:00</td>
      <td>Shake That</td>
      <td>Eminem</td>
      <td>6.512400</td>
      <td>0.963</td>
      <td>0.643</td>
      <td>0.534</td>
      <td>-5.785</td>
      <td>0.000049</td>
      <td>0.05070</td>
      <td>107.005</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4119</th>
      <td>2021-05-29 11:04:00</td>
      <td>Best Friend</td>
      <td>Foster The People</td>
      <td>1.916850</td>
      <td>0.959</td>
      <td>0.598</td>
      <td>0.408</td>
      <td>-5.534</td>
      <td>0.000000</td>
      <td>0.03580</td>
      <td>127.028</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4208</th>
      <td>2021-06-05 10:01:00</td>
      <td>Yard Man - Original</td>
      <td>Chris Michaels</td>
      <td>1.659050</td>
      <td>0.958</td>
      <td>0.799</td>
      <td>0.791</td>
      <td>-5.126</td>
      <td>0.000062</td>
      <td>0.02380</td>
      <td>127.985</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



Energy: to analyse if it is a track calm/peaceful or a track that will gives you some energy, gives you the urge to clap your hands, jump, run, ...
0: calm music / 1: High energy


```python
#Energy < 0.2: Quiet/peaceful/slow music (it's often piano like Debussy's Claire de Lune).  
#(Nostalgia and Shadows do not belong here).
spotify_features[(spotify_features['energy'] < 0.2)].drop_duplicates(subset = ['track', 'artist']).sort_values('energy').head(5)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>datetime</th>
      <th>track</th>
      <th>artist</th>
      <th>min_played</th>
      <th>danceability</th>
      <th>energy</th>
      <th>valence</th>
      <th>loudness</th>
      <th>instrumentalness</th>
      <th>acousticness</th>
      <th>tempo</th>
      <th>mode</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>746</th>
      <td>2020-08-14 20:42:00</td>
      <td>Nostalgia</td>
      <td>Ronnie Pacitti</td>
      <td>3.932650</td>
      <td>0.165</td>
      <td>0.00243</td>
      <td>0.0368</td>
      <td>-37.115</td>
      <td>0.875</td>
      <td>0.985</td>
      <td>68.420</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1961</th>
      <td>2020-08-31 09:44:00</td>
      <td>Shadows</td>
      <td>Talisco</td>
      <td>1.857733</td>
      <td>0.157</td>
      <td>0.00476</td>
      <td>0.0305</td>
      <td>-33.114</td>
      <td>0.955</td>
      <td>0.952</td>
      <td>131.936</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3635</th>
      <td>2021-04-16 12:11:00</td>
      <td>Claire de lune</td>
      <td>Claude Debussy</td>
      <td>2.534983</td>
      <td>0.365</td>
      <td>0.01000</td>
      <td>0.0364</td>
      <td>-25.268</td>
      <td>0.924</td>
      <td>0.995</td>
      <td>135.048</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3056</th>
      <td>2021-01-06 11:48:00</td>
      <td>This Way Or Another</td>
      <td>Owen Kennedy</td>
      <td>0.500783</td>
      <td>0.356</td>
      <td>0.02160</td>
      <td>0.1190</td>
      <td>-21.830</td>
      <td>0.899</td>
      <td>0.995</td>
      <td>100.084</td>
      <td>1</td>
    </tr>
    <tr>
      <th>581</th>
      <td>2020-08-13 11:28:00</td>
      <td>Overnight</td>
      <td>Chilly Gonzales</td>
      <td>3.380667</td>
      <td>0.388</td>
      <td>0.02440</td>
      <td>0.1890</td>
      <td>-25.245</td>
      <td>0.894</td>
      <td>0.991</td>
      <td>80.132</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
#Energy > 0.9: Music speed/gives you the urge to jump, clap, run,...
#(Ritmo Especial doesn't belong here)
spotify_features[(spotify_features['energy'] > 0.9)].drop_duplicates(subset = ['track', 'artist']).sort_values('energy', ascending= False).head(5)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>datetime</th>
      <th>track</th>
      <th>artist</th>
      <th>min_played</th>
      <th>danceability</th>
      <th>energy</th>
      <th>valence</th>
      <th>loudness</th>
      <th>instrumentalness</th>
      <th>acousticness</th>
      <th>tempo</th>
      <th>mode</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2428</th>
      <td>2020-10-30 12:08:00</td>
      <td>Ritmo Especial</td>
      <td>Daniel Maloso</td>
      <td>4.116833</td>
      <td>0.802</td>
      <td>0.996</td>
      <td>0.496</td>
      <td>-4.665</td>
      <td>0.861000</td>
      <td>0.00939</td>
      <td>120.996</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4200</th>
      <td>2021-06-05 09:44:00</td>
      <td>TURN OFF THE LIGHTS</td>
      <td>Dog Blood</td>
      <td>0.500383</td>
      <td>0.756</td>
      <td>0.994</td>
      <td>0.765</td>
      <td>-2.466</td>
      <td>0.717000</td>
      <td>0.00217</td>
      <td>128.027</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2974</th>
      <td>2020-12-28 21:09:00</td>
      <td>Hippy Hippy Shake</td>
      <td>Big Soul</td>
      <td>2.846367</td>
      <td>0.738</td>
      <td>0.994</td>
      <td>0.529</td>
      <td>-5.445</td>
      <td>0.000325</td>
      <td>0.04700</td>
      <td>129.307</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4567</th>
      <td>2021-07-08 17:17:00</td>
      <td>Two-Timing Touch and Broken Bones</td>
      <td>The Hives</td>
      <td>2.008433</td>
      <td>0.342</td>
      <td>0.992</td>
      <td>0.938</td>
      <td>-3.251</td>
      <td>0.006040</td>
      <td>0.01900</td>
      <td>165.230</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3179</th>
      <td>2021-02-04 21:03:00</td>
      <td>Banana Split</td>
      <td>Lio</td>
      <td>1.374283</td>
      <td>0.677</td>
      <td>0.985</td>
      <td>0.967</td>
      <td>-3.577</td>
      <td>0.005240</td>
      <td>0.06350</td>
      <td>156.141</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



Valence: to analyse if it is a track that will put you on a good or bad/melancholic mood.
0: bad/melancholic mood / 1: good mood.


```python
#Valence < 0.1: Music dark/bad mood/melancholic 
#(shadows doesn't belong here)
spotify_features[(spotify_features['valence'] < 0.2)].drop_duplicates(subset = ['track', 'artist']).sort_values('valence').head(5)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>datetime</th>
      <th>track</th>
      <th>artist</th>
      <th>min_played</th>
      <th>danceability</th>
      <th>energy</th>
      <th>valence</th>
      <th>loudness</th>
      <th>instrumentalness</th>
      <th>acousticness</th>
      <th>tempo</th>
      <th>mode</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>3172</th>
      <td>2021-01-30 15:30:00</td>
      <td>Ephos</td>
      <td>Flug</td>
      <td>7.872167</td>
      <td>0.719</td>
      <td>0.72700</td>
      <td>0.0296</td>
      <td>-10.563</td>
      <td>0.945000</td>
      <td>0.0423</td>
      <td>132.000</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4435</th>
      <td>2021-06-12 19:07:00</td>
      <td>Queens</td>
      <td>The Blaze</td>
      <td>2.217667</td>
      <td>0.617</td>
      <td>0.58900</td>
      <td>0.0301</td>
      <td>-12.767</td>
      <td>0.268000</td>
      <td>0.6900</td>
      <td>125.058</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1961</th>
      <td>2020-08-31 09:44:00</td>
      <td>Shadows</td>
      <td>Talisco</td>
      <td>1.857733</td>
      <td>0.157</td>
      <td>0.00476</td>
      <td>0.0305</td>
      <td>-33.114</td>
      <td>0.955000</td>
      <td>0.9520</td>
      <td>131.936</td>
      <td>1</td>
    </tr>
    <tr>
      <th>317</th>
      <td>2020-08-12 11:22:00</td>
      <td>Loreley</td>
      <td>Kölsch</td>
      <td>5.750000</td>
      <td>0.781</td>
      <td>0.50900</td>
      <td>0.0311</td>
      <td>-7.793</td>
      <td>0.000022</td>
      <td>0.2910</td>
      <td>127.964</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4252</th>
      <td>2021-06-07 15:36:00</td>
      <td>After Party</td>
      <td>French 79</td>
      <td>5.455100</td>
      <td>0.532</td>
      <td>0.57700</td>
      <td>0.0335</td>
      <td>-12.320</td>
      <td>0.905000</td>
      <td>0.1640</td>
      <td>119.041</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
#Valence > 0.9: Music good mood/happy
spotify_features[(spotify_features['valence'] > 0.9)].drop_duplicates(subset = ['track', 'artist']).sort_values('valence', ascending= False).head(5)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>datetime</th>
      <th>track</th>
      <th>artist</th>
      <th>min_played</th>
      <th>danceability</th>
      <th>energy</th>
      <th>valence</th>
      <th>loudness</th>
      <th>instrumentalness</th>
      <th>acousticness</th>
      <th>tempo</th>
      <th>mode</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>384</th>
      <td>2020-08-12 13:11:00</td>
      <td>Gotta Go Home</td>
      <td>Boney M.</td>
      <td>3.760433</td>
      <td>0.781</td>
      <td>0.936</td>
      <td>0.980</td>
      <td>-5.843</td>
      <td>0.052200</td>
      <td>0.2860</td>
      <td>131.659</td>
      <td>1</td>
    </tr>
    <tr>
      <th>934</th>
      <td>2020-08-15 17:31:00</td>
      <td>September</td>
      <td>Earth, Wind &amp; Fire</td>
      <td>3.584700</td>
      <td>0.697</td>
      <td>0.832</td>
      <td>0.979</td>
      <td>-7.264</td>
      <td>0.001310</td>
      <td>0.1680</td>
      <td>125.926</td>
      <td>1</td>
    </tr>
    <tr>
      <th>469</th>
      <td>2020-08-12 14:25:00</td>
      <td>Passe mon truc</td>
      <td>Stupeflip</td>
      <td>3.280217</td>
      <td>0.679</td>
      <td>0.876</td>
      <td>0.978</td>
      <td>-6.033</td>
      <td>0.149000</td>
      <td>0.0961</td>
      <td>160.459</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3202</th>
      <td>2021-02-05 21:20:00</td>
      <td>Je fume pu d'shit</td>
      <td>Stupeflip</td>
      <td>3.282000</td>
      <td>0.894</td>
      <td>0.544</td>
      <td>0.976</td>
      <td>-5.402</td>
      <td>0.004150</td>
      <td>0.0672</td>
      <td>107.292</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3259</th>
      <td>2021-02-20 21:07:00</td>
      <td>Pata Pata - Mono Version</td>
      <td>Miriam Makeba</td>
      <td>2.867800</td>
      <td>0.837</td>
      <td>0.853</td>
      <td>0.975</td>
      <td>-5.417</td>
      <td>0.000003</td>
      <td>0.6150</td>
      <td>126.845</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



Loudness: to analyse if the track is more or less loud. 
Range: -60 and 0 (db).


```python
#loudness < -20: Music like piano songs that are not loud
#(Nostalgia and shadows do not belong here)
spotify_features[(spotify_features['loudness'] < -20)].drop_duplicates(subset = ['track', 'artist']).sort_values('loudness').head(5)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>datetime</th>
      <th>track</th>
      <th>artist</th>
      <th>min_played</th>
      <th>danceability</th>
      <th>energy</th>
      <th>valence</th>
      <th>loudness</th>
      <th>instrumentalness</th>
      <th>acousticness</th>
      <th>tempo</th>
      <th>mode</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>746</th>
      <td>2020-08-14 20:42:00</td>
      <td>Nostalgia</td>
      <td>Ronnie Pacitti</td>
      <td>3.932650</td>
      <td>0.165</td>
      <td>0.00243</td>
      <td>0.0368</td>
      <td>-37.115</td>
      <td>0.875</td>
      <td>0.985</td>
      <td>68.420</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1961</th>
      <td>2020-08-31 09:44:00</td>
      <td>Shadows</td>
      <td>Talisco</td>
      <td>1.857733</td>
      <td>0.157</td>
      <td>0.00476</td>
      <td>0.0305</td>
      <td>-33.114</td>
      <td>0.955</td>
      <td>0.952</td>
      <td>131.936</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3635</th>
      <td>2021-04-16 12:11:00</td>
      <td>Claire de lune</td>
      <td>Claude Debussy</td>
      <td>2.534983</td>
      <td>0.365</td>
      <td>0.01000</td>
      <td>0.0364</td>
      <td>-25.268</td>
      <td>0.924</td>
      <td>0.995</td>
      <td>135.048</td>
      <td>1</td>
    </tr>
    <tr>
      <th>581</th>
      <td>2020-08-13 11:28:00</td>
      <td>Overnight</td>
      <td>Chilly Gonzales</td>
      <td>3.380667</td>
      <td>0.388</td>
      <td>0.02440</td>
      <td>0.1890</td>
      <td>-25.245</td>
      <td>0.894</td>
      <td>0.991</td>
      <td>80.132</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1271</th>
      <td>2020-08-16 15:45:00</td>
      <td>The Entertainer</td>
      <td>Liberace</td>
      <td>2.045517</td>
      <td>0.466</td>
      <td>0.15000</td>
      <td>0.7960</td>
      <td>-24.581</td>
      <td>0.877</td>
      <td>0.992</td>
      <td>159.517</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
#Loudness > -5 : Music like rock songs that are loud
spotify_features[(spotify_features['loudness'] > -5)].drop_duplicates(subset = ['track', 'artist']).sort_values('loudness', ascending = False).head(5)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>datetime</th>
      <th>track</th>
      <th>artist</th>
      <th>min_played</th>
      <th>danceability</th>
      <th>energy</th>
      <th>valence</th>
      <th>loudness</th>
      <th>instrumentalness</th>
      <th>acousticness</th>
      <th>tempo</th>
      <th>mode</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>205</th>
      <td>2020-08-12 09:28:00</td>
      <td>Menez daou</td>
      <td>Les Ramoneurs De Menhirs</td>
      <td>5.162217</td>
      <td>0.442</td>
      <td>0.946</td>
      <td>0.660</td>
      <td>0.074</td>
      <td>0.00767</td>
      <td>0.09380</td>
      <td>177.137</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4357</th>
      <td>2021-06-09 12:06:00</td>
      <td>Poundshop Kardashians</td>
      <td>Sam Fender</td>
      <td>2.654917</td>
      <td>0.557</td>
      <td>0.875</td>
      <td>0.770</td>
      <td>-1.596</td>
      <td>0.00000</td>
      <td>0.04190</td>
      <td>138.023</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4338</th>
      <td>2021-06-09 10:41:00</td>
      <td>Bear Claws</td>
      <td>The Academic</td>
      <td>3.568633</td>
      <td>0.552</td>
      <td>0.877</td>
      <td>0.628</td>
      <td>-1.879</td>
      <td>0.00000</td>
      <td>0.01420</td>
      <td>97.056</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2956</th>
      <td>2020-12-28 20:38:00</td>
      <td>Louxor J'Adore - Katerine vs Joachim Garraud</td>
      <td>Philippe Katerine</td>
      <td>3.118433</td>
      <td>0.471</td>
      <td>0.929</td>
      <td>0.696</td>
      <td>-1.897</td>
      <td>0.49600</td>
      <td>0.00217</td>
      <td>134.717</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4198</th>
      <td>2021-06-05 09:42:00</td>
      <td>BREAK LAW</td>
      <td>Dog Blood</td>
      <td>0.512767</td>
      <td>0.679</td>
      <td>0.950</td>
      <td>0.120</td>
      <td>-2.247</td>
      <td>0.00205</td>
      <td>0.01480</td>
      <td>107.994</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



Instrumentalness: to analyse if a track has vocals or not.
0: A lots of vocals / 1: no vocals.


```python
#Instru 0.9 - 1: no vocals at all.
#(shadows doesn't belong here)
spotify_features[(spotify_features['instrumentalness'] > 0.5)].drop_duplicates(subset = ['track', 'artist']).sort_values('instrumentalness', ascending=False).head(5)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>datetime</th>
      <th>track</th>
      <th>artist</th>
      <th>min_played</th>
      <th>danceability</th>
      <th>energy</th>
      <th>valence</th>
      <th>loudness</th>
      <th>instrumentalness</th>
      <th>acousticness</th>
      <th>tempo</th>
      <th>mode</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>4639</th>
      <td>2021-07-11 12:17:00</td>
      <td>Bilboquet (Sirba)</td>
      <td>Polo &amp; Pan</td>
      <td>3.269100</td>
      <td>0.803</td>
      <td>0.78500</td>
      <td>0.3270</td>
      <td>-7.757</td>
      <td>0.962</td>
      <td>0.483</td>
      <td>100.028</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3588</th>
      <td>2021-04-13 17:30:00</td>
      <td>Where Is My Mind</td>
      <td>Maxence Cyrin</td>
      <td>2.752667</td>
      <td>0.333</td>
      <td>0.11400</td>
      <td>0.0566</td>
      <td>-21.255</td>
      <td>0.960</td>
      <td>0.906</td>
      <td>141.839</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3629</th>
      <td>2021-04-16 11:29:00</td>
      <td>Walk to School</td>
      <td>Philip Glass</td>
      <td>1.742283</td>
      <td>0.225</td>
      <td>0.03680</td>
      <td>0.0388</td>
      <td>-23.039</td>
      <td>0.956</td>
      <td>0.977</td>
      <td>169.851</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1961</th>
      <td>2020-08-31 09:44:00</td>
      <td>Shadows</td>
      <td>Talisco</td>
      <td>1.857733</td>
      <td>0.157</td>
      <td>0.00476</td>
      <td>0.0305</td>
      <td>-33.114</td>
      <td>0.955</td>
      <td>0.952</td>
      <td>131.936</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3023</th>
      <td>2020-12-31 17:14:00</td>
      <td>Hip Hop First of All</td>
      <td>Guts</td>
      <td>2.675700</td>
      <td>0.652</td>
      <td>0.54900</td>
      <td>0.1660</td>
      <td>-9.090</td>
      <td>0.950</td>
      <td>0.218</td>
      <td>83.007</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
#Instru 0.5 - 0.9: Almost no voice. 
#Mostly it's techno/electro music with voice samples as in CamelPhat's Cola. 
#Or a song with just a few backing vocals like l'estasi dell'oro.
spotify_features[(spotify_features['instrumentalness'] > 0.5)].drop_duplicates(subset = ['track', 'artist']).sort_values('instrumentalness').head(5)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>datetime</th>
      <th>track</th>
      <th>artist</th>
      <th>min_played</th>
      <th>danceability</th>
      <th>energy</th>
      <th>valence</th>
      <th>loudness</th>
      <th>instrumentalness</th>
      <th>acousticness</th>
      <th>tempo</th>
      <th>mode</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>869</th>
      <td>2020-08-15 14:18:00</td>
      <td>Cola</td>
      <td>CamelPhat</td>
      <td>3.728650</td>
      <td>0.706</td>
      <td>0.740</td>
      <td>0.444</td>
      <td>-7.904</td>
      <td>0.512</td>
      <td>0.02450</td>
      <td>122.007</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3531</th>
      <td>2021-04-09 12:25:00</td>
      <td>L'estasi dell'oro</td>
      <td>Ennio Morricone</td>
      <td>3.384433</td>
      <td>0.136</td>
      <td>0.484</td>
      <td>0.102</td>
      <td>-12.945</td>
      <td>0.523</td>
      <td>0.71500</td>
      <td>99.566</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3467</th>
      <td>2021-03-27 21:49:00</td>
      <td>Salam Aleykoum</td>
      <td>Salut C'est Cool</td>
      <td>5.450317</td>
      <td>0.690</td>
      <td>0.963</td>
      <td>0.154</td>
      <td>-6.341</td>
      <td>0.527</td>
      <td>0.00776</td>
      <td>138.001</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2158</th>
      <td>2020-10-14 17:24:00</td>
      <td>Pigalle</td>
      <td>Bellaire</td>
      <td>4.169933</td>
      <td>0.906</td>
      <td>0.661</td>
      <td>0.228</td>
      <td>-7.532</td>
      <td>0.527</td>
      <td>0.00417</td>
      <td>124.987</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2606</th>
      <td>2020-12-09 11:57:00</td>
      <td>Enoi - Âme Live Version</td>
      <td>Âme</td>
      <td>0.727617</td>
      <td>0.709</td>
      <td>0.736</td>
      <td>0.389</td>
      <td>-9.095</td>
      <td>0.531</td>
      <td>0.00849</td>
      <td>124.407</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
#Instru 0.1 - 0.5: Music with more vocals but still a lot of techno/electro.
spotify_features[(spotify_features['instrumentalness'] > 0.1)].drop_duplicates(subset = ['track', 'artist']).sort_values('instrumentalness').head(5)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>datetime</th>
      <th>track</th>
      <th>artist</th>
      <th>min_played</th>
      <th>danceability</th>
      <th>energy</th>
      <th>valence</th>
      <th>loudness</th>
      <th>instrumentalness</th>
      <th>acousticness</th>
      <th>tempo</th>
      <th>mode</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>4424</th>
      <td>2021-06-12 18:14:00</td>
      <td>Don't Wanna Dance</td>
      <td>Boston Bun</td>
      <td>2.823367</td>
      <td>0.797</td>
      <td>0.827</td>
      <td>0.680</td>
      <td>-5.607</td>
      <td>0.101</td>
      <td>0.03340</td>
      <td>121.979</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3119</th>
      <td>2021-01-21 04:49:00</td>
      <td>Dr. Greenthumb</td>
      <td>Cypress Hill</td>
      <td>3.154217</td>
      <td>0.803</td>
      <td>0.547</td>
      <td>0.144</td>
      <td>-8.779</td>
      <td>0.102</td>
      <td>0.04820</td>
      <td>103.853</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3323</th>
      <td>2021-03-13 04:04:00</td>
      <td>Bout de bois</td>
      <td>Salut C'est Cool</td>
      <td>3.447933</td>
      <td>0.702</td>
      <td>0.758</td>
      <td>0.134</td>
      <td>-10.740</td>
      <td>0.104</td>
      <td>0.00295</td>
      <td>165.963</td>
      <td>1</td>
    </tr>
    <tr>
      <th>9</th>
      <td>2020-07-17 09:32:00</td>
      <td>Le métro et le bus</td>
      <td>Lewis OfMan</td>
      <td>0.925917</td>
      <td>0.592</td>
      <td>0.576</td>
      <td>0.276</td>
      <td>-8.279</td>
      <td>0.107</td>
      <td>0.49600</td>
      <td>110.045</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1604</th>
      <td>2020-08-21 10:33:00</td>
      <td>Il fait chaud</td>
      <td>Corine</td>
      <td>2.294550</td>
      <td>0.679</td>
      <td>0.834</td>
      <td>0.453</td>
      <td>-6.947</td>
      <td>0.108</td>
      <td>0.02100</td>
      <td>108.000</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
#Instru 0 - 0.1 : Songs with a lot of vocals.
spotify_features[(spotify_features['instrumentalness'] < 0.1)].drop_duplicates(subset = ['track', 'artist']).sort_values('instrumentalness').head(5)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>datetime</th>
      <th>track</th>
      <th>artist</th>
      <th>min_played</th>
      <th>danceability</th>
      <th>energy</th>
      <th>valence</th>
      <th>loudness</th>
      <th>instrumentalness</th>
      <th>acousticness</th>
      <th>tempo</th>
      <th>mode</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>3054</th>
      <td>2020-12-31 19:34:00</td>
      <td>Chérie</td>
      <td>Amadou &amp; Mariam</td>
      <td>1.943750</td>
      <td>0.834</td>
      <td>0.964</td>
      <td>0.680</td>
      <td>-3.822</td>
      <td>0.0</td>
      <td>0.1980</td>
      <td>127.994</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3126</th>
      <td>2021-01-30 04:38:00</td>
      <td>Les prisons de Nantes</td>
      <td>Tri Yann</td>
      <td>2.345400</td>
      <td>0.528</td>
      <td>0.395</td>
      <td>0.962</td>
      <td>-10.130</td>
      <td>0.0</td>
      <td>0.8040</td>
      <td>161.460</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3065</th>
      <td>2021-01-14 23:26:00</td>
      <td>Is This Love</td>
      <td>Bob Marley &amp; The Wailers</td>
      <td>3.845550</td>
      <td>0.776</td>
      <td>0.559</td>
      <td>0.758</td>
      <td>-8.375</td>
      <td>0.0</td>
      <td>0.1100</td>
      <td>122.242</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3063</th>
      <td>2021-01-14 23:22:00</td>
      <td>One Love / People Get Ready - Medley</td>
      <td>Bob Marley &amp; The Wailers</td>
      <td>2.882217</td>
      <td>0.725</td>
      <td>0.523</td>
      <td>0.950</td>
      <td>-9.593</td>
      <td>0.0</td>
      <td>0.0783</td>
      <td>76.292</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3055</th>
      <td>2021-01-03 18:34:00</td>
      <td>You Really Got Me - Mono Mix</td>
      <td>The Kinks</td>
      <td>2.237717</td>
      <td>0.573</td>
      <td>0.939</td>
      <td>0.963</td>
      <td>-6.441</td>
      <td>0.0</td>
      <td>0.4930</td>
      <td>137.382</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



Acousticness: to analyse if the track is more or less acoustic.
0: not accoustic / 1: 100% acoustic.


```python
#Acousticness < 0.2 : Not acoustic music (like band music or electro/techno)
spotify_features[(spotify_features['acousticness'] < 0.2)].drop_duplicates(subset = ['track', 'artist']).sort_values('acousticness').head(5)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>datetime</th>
      <th>track</th>
      <th>artist</th>
      <th>min_played</th>
      <th>danceability</th>
      <th>energy</th>
      <th>valence</th>
      <th>loudness</th>
      <th>instrumentalness</th>
      <th>acousticness</th>
      <th>tempo</th>
      <th>mode</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>4569</th>
      <td>2021-07-08 17:23:00</td>
      <td>Good Samaritan</td>
      <td>The Hives</td>
      <td>3.115000</td>
      <td>0.186</td>
      <td>0.924</td>
      <td>0.2570</td>
      <td>-3.830</td>
      <td>0.00968</td>
      <td>0.000002</td>
      <td>149.966</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4231</th>
      <td>2021-06-05 10:15:00</td>
      <td>Robot Rock</td>
      <td>Daft Punk</td>
      <td>0.802233</td>
      <td>0.590</td>
      <td>0.787</td>
      <td>0.5980</td>
      <td>-5.766</td>
      <td>0.84500</td>
      <td>0.000007</td>
      <td>111.926</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1510</th>
      <td>2020-08-17 18:29:00</td>
      <td>Hellifornia</td>
      <td>Gesaffelstein</td>
      <td>1.349033</td>
      <td>0.537</td>
      <td>0.784</td>
      <td>0.0384</td>
      <td>-3.734</td>
      <td>0.28800</td>
      <td>0.000009</td>
      <td>93.988</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4371</th>
      <td>2021-06-09 12:57:00</td>
      <td>Mixtape 2003</td>
      <td>The Academic</td>
      <td>3.404167</td>
      <td>0.288</td>
      <td>0.933</td>
      <td>0.3450</td>
      <td>-4.120</td>
      <td>0.33300</td>
      <td>0.000013</td>
      <td>167.109</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3625</th>
      <td>2021-04-15 18:19:00</td>
      <td>Force majeure</td>
      <td>Gaspard Augé</td>
      <td>3.435283</td>
      <td>0.578</td>
      <td>0.696</td>
      <td>0.2190</td>
      <td>-6.866</td>
      <td>0.87400</td>
      <td>0.000048</td>
      <td>119.999</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
#Acousticness > 0.9 : Highly acoustic music (like piano)
spotify_features[(spotify_features['acousticness'] > 0.9)].drop_duplicates(subset = ['track', 'artist']).sort_values('acousticness', ascending= False).head(5)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>datetime</th>
      <th>track</th>
      <th>artist</th>
      <th>min_played</th>
      <th>danceability</th>
      <th>energy</th>
      <th>valence</th>
      <th>loudness</th>
      <th>instrumentalness</th>
      <th>acousticness</th>
      <th>tempo</th>
      <th>mode</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>3635</th>
      <td>2021-04-16 12:11:00</td>
      <td>Claire de lune</td>
      <td>Claude Debussy</td>
      <td>2.534983</td>
      <td>0.365</td>
      <td>0.0100</td>
      <td>0.0364</td>
      <td>-25.268</td>
      <td>0.924</td>
      <td>0.995</td>
      <td>135.048</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3056</th>
      <td>2021-01-06 11:48:00</td>
      <td>This Way Or Another</td>
      <td>Owen Kennedy</td>
      <td>0.500783</td>
      <td>0.356</td>
      <td>0.0216</td>
      <td>0.1190</td>
      <td>-21.830</td>
      <td>0.899</td>
      <td>0.995</td>
      <td>100.084</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1271</th>
      <td>2020-08-16 15:45:00</td>
      <td>The Entertainer</td>
      <td>Liberace</td>
      <td>2.045517</td>
      <td>0.466</td>
      <td>0.1500</td>
      <td>0.7960</td>
      <td>-24.581</td>
      <td>0.877</td>
      <td>0.992</td>
      <td>159.517</td>
      <td>1</td>
    </tr>
    <tr>
      <th>581</th>
      <td>2020-08-13 11:28:00</td>
      <td>Overnight</td>
      <td>Chilly Gonzales</td>
      <td>3.380667</td>
      <td>0.388</td>
      <td>0.0244</td>
      <td>0.1890</td>
      <td>-25.245</td>
      <td>0.894</td>
      <td>0.991</td>
      <td>80.132</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3630</th>
      <td>2021-04-16 11:35:00</td>
      <td>Gaze</td>
      <td>Moux</td>
      <td>2.669333</td>
      <td>0.574</td>
      <td>0.0577</td>
      <td>0.0849</td>
      <td>-22.165</td>
      <td>0.912</td>
      <td>0.991</td>
      <td>140.994</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



Ok, so we have to drop the outliers that have ended up in places where they don't belong, to get better results. It's like the tempo, sometimes the recognition is wrong because of certain elements of the music. 


```python
#Let's drop those records and validate the drop with .shape and assert
print(spotify_features.shape)
spotify_features = spotify_features[spotify_features['track'] != 'Shadows']
spotify_features = spotify_features[spotify_features['track'] != 'Nostalgia']
spotify_features = spotify_features[spotify_features['track'] != 'Awake']
spotify_features = spotify_features[spotify_features['track'] != 'Quick Drive']
spotify_features = spotify_features[spotify_features['track'] != 'Liquid sunshine']
spotify_features = spotify_features[spotify_features['track'] != 'Ritmo Especial']
print(spotify_features.shape)
assert spotify_features[(spotify_features['track'] == 'Shadows')| (spotify_features['track'] == 'Nostalgia') | (spotify_features['track'] == 'Awake')].empty

```

    (4652, 12)
    (4638, 12)
    

#### Rescaling the audio features for comparison: 

To be able to compare the features, we need to convert them to the same scale. We can use the min max scaling technique to convert their values between 0 and 1.
(They are all already between 0 and 1, except for the loudness, so I prefer to scale them back.)


```python
from sklearn.preprocessing import MinMaxScaler
#Let's rescale the features
min_max_scaler_tt = MinMaxScaler()

spotify_features.iloc[:,4:10]=min_max_scaler_tt.fit_transform(spotify_features.iloc[:,4:10])
spotify_features.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>datetime</th>
      <th>track</th>
      <th>artist</th>
      <th>min_played</th>
      <th>danceability</th>
      <th>energy</th>
      <th>valence</th>
      <th>loudness</th>
      <th>instrumentalness</th>
      <th>acousticness</th>
      <th>tempo</th>
      <th>mode</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2020-07-16 15:43:00</td>
      <td>L'Orchestrina</td>
      <td>Paolo Conte</td>
      <td>3.314000</td>
      <td>0.739596</td>
      <td>0.822154</td>
      <td>0.755892</td>
      <td>0.653855</td>
      <td>0.000191</td>
      <td>0.580904</td>
      <td>117.47</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2020-08-15 19:14:00</td>
      <td>L'Orchestrina</td>
      <td>Paolo Conte</td>
      <td>3.311583</td>
      <td>0.739596</td>
      <td>0.822154</td>
      <td>0.755892</td>
      <td>0.653855</td>
      <td>0.000191</td>
      <td>0.580904</td>
      <td>117.47</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2020-08-17 13:01:00</td>
      <td>L'Orchestrina</td>
      <td>Paolo Conte</td>
      <td>3.311533</td>
      <td>0.739596</td>
      <td>0.822154</td>
      <td>0.755892</td>
      <td>0.653855</td>
      <td>0.000191</td>
      <td>0.580904</td>
      <td>117.47</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2020-08-23 15:41:00</td>
      <td>L'Orchestrina</td>
      <td>Paolo Conte</td>
      <td>3.311450</td>
      <td>0.739596</td>
      <td>0.822154</td>
      <td>0.755892</td>
      <td>0.653855</td>
      <td>0.000191</td>
      <td>0.580904</td>
      <td>117.47</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2020-08-25 14:27:00</td>
      <td>L'Orchestrina</td>
      <td>Paolo Conte</td>
      <td>3.314000</td>
      <td>0.739596</td>
      <td>0.822154</td>
      <td>0.755892</td>
      <td>0.653855</td>
      <td>0.000191</td>
      <td>0.580904</td>
      <td>117.47</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
spotify_features.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>min_played</th>
      <th>danceability</th>
      <th>energy</th>
      <th>valence</th>
      <th>loudness</th>
      <th>instrumentalness</th>
      <th>acousticness</th>
      <th>tempo</th>
      <th>mode</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>4638.000000</td>
      <td>4638.000000</td>
      <td>4638.000000</td>
      <td>4638.000000</td>
      <td>4638.000000</td>
      <td>4638.000000</td>
      <td>4638.000000</td>
      <td>4638.000000</td>
      <td>4638.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>3.488633</td>
      <td>0.646954</td>
      <td>0.634980</td>
      <td>0.572766</td>
      <td>0.678102</td>
      <td>0.173087</td>
      <td>0.264564</td>
      <td>116.959213</td>
      <td>0.581716</td>
    </tr>
    <tr>
      <th>std</th>
      <td>1.507279</td>
      <td>0.176804</td>
      <td>0.189612</td>
      <td>0.271474</td>
      <td>0.117976</td>
      <td>0.296806</td>
      <td>0.276047</td>
      <td>24.042351</td>
      <td>0.493330</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.500017</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>58.583000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>2.737333</td>
      <td>0.536266</td>
      <td>0.519309</td>
      <td>0.359217</td>
      <td>0.616151</td>
      <td>0.000013</td>
      <td>0.033063</td>
      <td>99.822250</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>3.469892</td>
      <td>0.673603</td>
      <td>0.659553</td>
      <td>0.594907</td>
      <td>0.690040</td>
      <td>0.002401</td>
      <td>0.157787</td>
      <td>117.010000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>4.216121</td>
      <td>0.774078</td>
      <td>0.782520</td>
      <td>0.806397</td>
      <td>0.760398</td>
      <td>0.212058</td>
      <td>0.438943</td>
      <td>127.982000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>30.725217</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>210.164000</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>



Ok, now let's analyse the correlation between features

#### Correlation matrix of the audio features: <a class="anchor" id="section_3_2_2"></a>


```python
sns.heatmap(round(spotify_features.corr(),1),vmin=-1, vmax=1, annot = True,
            cbar_kws={'label': '1 = Positively correlated\n -1 = Negatively correlated'})
plt.title('Correlation between features')
plt.show()
```


![corrmatrix_png]( https://github.com/TristanT56/My-Spotify-Data-Analysis---Python/blob/main/Images%20for%20Readme%20markdown/output_88_0.png)
    


**Correlation results:**

1 - Valence is a bit positively correlated (0.4) with danceability and energy. If the music scores high on danceability or energy, it is more likely to score high on valence as well. In other words, if music makes you want to jump (energy) or dance (danceability), it will also put you in a good mood (valence).

2 - Loudness is quite positively correlated (0.7) with energy. If the music scores high on loudness, it is more likely to score high high on energy as well. In other words, more the music is loud, more it gives you the urge to jump/clap (energy).

3 - Acousticness is somehow negatively correlated (-0.6) with energy. If music scores high on acousticness, it is more likely to score low on energy. In other words, the more acoustic the music, the less likely it is to make you want to jump/clap (energy). 

4 - Therefore, Acousticness is also a bit negatively correlated (-0.4) with loudness. If music scores high on acousticness, it is more likely to score low on loudness. In other words, less the music will be acoustic (like a rock band), louder it is more likely to be. 

So, for example, if you are listening to 100% piano music (very acoustic), it is more likely that the music does not have much energy and therefore not much valence. In addition, this music should not be too loud. At the end, this piano music is more likely to be quiet/peaceful/slow and should put you in a somewhat melancholic mood. 

#### Histograms and Boxplots - Part 2 - Understanding my audio features preferences: <a class="anchor" id="section_3_2_3"></a>

Let's replot the histograms and boxplots now that we have eliminated some weird outliers:


```python

fig, ax = plt.subplots(2, 3)

boxdistplot(spotify_features['danceability'], ax[0, 0])
boxdistplot(spotify_features['energy'], ax[0, 1])
boxdistplot(spotify_features['valence'], ax[0, 2])
boxdistplot(spotify_features['loudness'], ax[1, 0])
boxdistplot(spotify_features['instrumentalness'], ax[1, 1])
boxdistplot(spotify_features['acousticness'], ax[1, 2])

plt.subplots_adjust(right=2, top= 1.5 , wspace=0.3, hspace=0.3)
plt.show()
```

![EDA2table21_png]( https://github.com/TristanT56/My-Spotify-Data-Analysis---Python/blob/main/Images%20for%20Readme%20markdown/output_92_0.png)
    



```python
fig, ax = plt.subplots(1, 2, squeeze=False)

boxdistplot(spotify_features['tempo'], ax[0, 0])
sns.histplot(spotify_features, x='mode', hue='mode',  ax=ax[0, 1])
#spotify_features['mode'].loc[spotify_features['mode']==1].count()
perc = spotify_features['mode'].value_counts(normalize=True).mul(100).round(1).astype(str) + '%'
ax[0, 1].legend( ['major = '+ perc.iloc[0], 'minor = '+ perc.iloc[1]], title='Mode:', loc="upper center")


plt.subplots_adjust(right=1.5, top= 0.75 , wspace=0.3, hspace=0.3)
plt.show()
```

![EDA2table22_png]( https://github.com/TristanT56/My-Spotify-Data-Analysis---Python/blob/main/Images%20for%20Readme%20markdown/output_93_0.png)
    


**My audio features results:** <a class="anchor" id="eda"></a>

Danceability: My musical tastes in terms of danceability are diverse (std = 0.18) but it seems that I have a small preference for music that is quite danceable (mean = 0.65).

Energy: As for dancability, my musical tastes in terms of energy are diversified (std = 0.19) and I have a small preference for music that has some energy (mean = 0.63).

Valence: Regarding valence, my musical tastes are very diversified (std = 0.27 and mean = 0.57). We can see two peaks/modes which indicate that I like music with a high valence score around 0.8 but also calm/melancholic music with a valence score around 0.45. (You can even see that I sometimes like to listen to music with a very low valence score. This is for example dark techno/electro music like 'Gesaffelstein').

Loudness: My musical taste in terms of loudness is not very diverse (std = 0.12), I like music that is quite loud (mean = 0.68).

Instrumentalness: My data in terms of instrumentalness are highly right-skewed (median = 0, mean = 0.17 and std = 0.3).  This indicates that the majority of my music has an instrumentality score of 0 (median=0) but as it is very diverse (std = 0.3 and mean = 0.17), we can also find some music with a very high instrumentality score. In other words, this means that I prefer real song/music with vocals (instrumentalness = 0) but I also like techo/electro music without any vocals like 'Bilboquet (Sirba)' from 'Polo & Pan' or 100% piano music (without vocals).

Acousticness: My data in terms of acousticness are also right-skewed (median = 0.16, mean = 0.26 and std = 0.28).  This indicates that the majority of my music has a low acousticness score (median=0.16) but as it is diverse (std = 0.28 and mean = 0.26), we can also find some music with a very high acousticness score. In other words, this means that I prefer real band music or electro/techno music (acousticness < 0.2) but I also like acoustic and quiet music like piano.

Tempo: As we said before, we have to take this tempo analysis very carefully as Spotify's API seems to have a lot of errors in tempo recognition. But to give a general idea, it seems I like all types of tempo, but I have a preference for tempo around 117 BPM (std = 24, mean and median = 117).

Mode: I seem to prefer music in major mode (58.2% of all my music). But since it's not a big difference, it means that I don't really care about the mode of the music.

So, to sum up: My musical tastes are very diverse. I like all types of tempo and mode, although I have a slight preference for music in major mode with a tempo around 117BPM. I listen to music that is more or less danceable, music with a lot of energy but also sometime calm music, music with a good valence but also music that is more melancholic, chill or even dark sometimes. I like loud music and I prefer music with voices that are not acoustic. However, I also sometimes like music without vocals to focus on the different sounds like electro/techno music or acoustic music that is calm/chill like piano music.


```python
#Ok, we will save that and start the analysis:
#let's save it into a new csv file:
spotify_features.to_csv(r'C:\Users\Tristan\Documents\DATA\spotify_project\spotify_features_clean.csv', index=False)
```

## C) Analysis Part 1 - Tops<a class="anchor" id="chapter4"></a>

In this first part of the analysis, my objectives are :
- To know which are my top artists and my top tracks.
- To see my music consumption on spotify per month during the analysis period (2020-07/2021-07).
- To see when I listen to the most spotify during the week (by day of the week and by hour).

For this part, we will use the first table 'spotify_tt' and not the full table with the features because 'spotify_tt' contains all my historical data, even those for which the spotify API could not find any information. As a reminder, we had to delete 85 records after we merged the two tables because they were tracks from spotify_tt for which the api could not find any features.


```python
#We need to set datetime as index
spotify_tt.set_index('datetime', inplace = True)
spotify_tt.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>track</th>
      <th>artist</th>
      <th>min_played</th>
    </tr>
    <tr>
      <th>datetime</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2020-07-16 15:43:00</th>
      <td>L'Orchestrina</td>
      <td>Paolo Conte</td>
      <td>3.314000</td>
    </tr>
    <tr>
      <th>2020-07-17 09:29:00</th>
      <td>fancy</td>
      <td>Isaac Delusion</td>
      <td>0.531017</td>
    </tr>
    <tr>
      <th>2020-07-17 09:31:00</th>
      <td>Plein de bisous</td>
      <td>Lewis OfMan</td>
      <td>1.056267</td>
    </tr>
    <tr>
      <th>2020-07-17 09:32:00</th>
      <td>Le métro et le bus</td>
      <td>Lewis OfMan</td>
      <td>0.925917</td>
    </tr>
    <tr>
      <th>2020-07-17 09:36:00</th>
      <td>La légende urbaine</td>
      <td>Voyou</td>
      <td>3.779767</td>
    </tr>
  </tbody>
</table>
</div>



### 1 - Top artists: <a class="anchor" id="section_4_1"></a>


```python
#Top 10 artist by nb of plays:
top10_artist_count = spotify_tt['artist'].value_counts().reset_index().head(10)
top10_artist_count.columns = ['artist', 'nb_of_plays']
top10_artist_count
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>artist</th>
      <th>nb_of_plays</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>The Lumineers</td>
      <td>309</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Jungle</td>
      <td>125</td>
    </tr>
    <tr>
      <th>2</th>
      <td>La Femme</td>
      <td>107</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Guts</td>
      <td>95</td>
    </tr>
    <tr>
      <th>4</th>
      <td>L'Impératrice</td>
      <td>85</td>
    </tr>
    <tr>
      <th>5</th>
      <td>easy life</td>
      <td>79</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Parcels</td>
      <td>78</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Lil Dicky</td>
      <td>77</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Polo &amp; Pan</td>
      <td>69</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Kid Francescoli</td>
      <td>68</td>
    </tr>
  </tbody>
</table>
</div>




```python
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_style("white")
sns.barplot(x= 'artist', y= 'nb_of_plays', data= top10_artist_count, palette='winter_r')
plt.xticks(rotation=70,  ha="right", rotation_mode="anchor")
plt.xlabel(None)
plt.ylabel('Nb of plays\n')
plt.title('Top 10 artists (Nb of plays)')

top10_artist_count['P'] = top10_artist_count.nb_of_plays.astype('str')
plt.legend( top10_artist_count['artist']+ '  (played '+ top10_artist_count['P'] +' times)',loc = 2, bbox_to_anchor = (1,1))

plt.savefig(r'C:\Users\Tristan\Documents\DATA\spotify_project\topartist1.png', bbox_inches='tight')
plt.show()

```


![artists1_png]( https://github.com/TristanT56/My-Spotify-Data-Analysis---Python/blob/main/Images%20for%20Readme%20markdown/output_101_0.png)
    



```python
#Top 10 artist by nb of hours:
top10_artist_h = spotify_tt.groupby('artist')['min_played'].sum().div(60).round(1).sort_values(ascending=False).reset_index().head(10)
top10_artist_h.columns = ['artist', 'nb_of_hours']
top10_artist_h
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>artist</th>
      <th>nb_of_hours</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>The Lumineers</td>
      <td>16.9</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Jungle</td>
      <td>6.9</td>
    </tr>
    <tr>
      <th>2</th>
      <td>La Femme</td>
      <td>6.3</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Guts</td>
      <td>6.1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>L'Impératrice</td>
      <td>5.6</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Lil Dicky</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Parcels</td>
      <td>4.9</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Polo &amp; Pan</td>
      <td>4.8</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Isaac Delusion</td>
      <td>4.5</td>
    </tr>
    <tr>
      <th>9</th>
      <td>easy life</td>
      <td>3.9</td>
    </tr>
  </tbody>
</table>
</div>




```python
sns.set_style("white")
sns.barplot(x= 'artist', y= 'nb_of_hours', data= top10_artist_h, palette='winter_r')
plt.xticks(rotation=70,  ha="right", rotation_mode="anchor")
plt.xlabel(None)
plt.ylabel('Nb of hours\n')
plt.title('Top 10 artists (Nb of Hours)')

top10_artist_h['H'] = top10_artist_h.nb_of_hours.astype('str')
plt.legend( top10_artist_h['artist']+ '  ('+ top10_artist_h['H'] +'H)',  loc = 2, bbox_to_anchor = (1,1))

plt.savefig(r'C:\Users\Tristan\Documents\DATA\spotify_project\topartist2.png', bbox_inches='tight')
plt.show()
```


![artists2_png]( https://github.com/TristanT56/My-Spotify-Data-Analysis---Python/blob/main/Images%20for%20Readme%20markdown/output_103_0.png)
    


### 2 - Top tracks: <a class="anchor" id="section_4_2"></a>


```python
#Top 10 track by nb of plays:
top10_track_count = spotify_tt[['track', 'artist']].value_counts().reset_index().head(10)
top10_track_count.columns = ['track','artist' ,'nb_of_plays']
top10_track_count
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>track</th>
      <th>artist</th>
      <th>nb_of_plays</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Disco Inferno</td>
      <td>The Trammps</td>
      <td>40</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Sledgehammer</td>
      <td>Peter Gabriel</td>
      <td>33</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Nomalizo</td>
      <td>Letta Mbulu</td>
      <td>32</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Nothing But A Heartache</td>
      <td>The Flirtations</td>
      <td>31</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Last Train to London</td>
      <td>Electric Light Orchestra</td>
      <td>30</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Isabella</td>
      <td>Isaac Delusion</td>
      <td>28</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Idol</td>
      <td>Mind Enterprises</td>
      <td>27</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Baianá</td>
      <td>Bakermat</td>
      <td>27</td>
    </tr>
    <tr>
      <th>8</th>
      <td>(More and More) It Ain't Easy</td>
      <td>Jungle</td>
      <td>27</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Maryland</td>
      <td>Elephanz</td>
      <td>26</td>
    </tr>
  </tbody>
</table>
</div>




```python
sns.set_style("white")
sns.barplot(x= 'track', y= 'nb_of_plays', data= top10_track_count, palette='winter_r')
plt.xticks(rotation=70,  ha="right", rotation_mode="anchor")
plt.xlabel(None)
plt.ylabel('Nb of plays\n')
plt.title('Top 10 tracks (Nb of plays)')

top10_track_count['P'] = top10_track_count.nb_of_plays.astype('str')
plt.legend(top10_track_count['track'] + ' - ' + top10_track_count['artist']+ '  (played '+ top10_track_count['P'] +' times)',  loc = 2, bbox_to_anchor = (1,1))

plt.savefig(r'C:\Users\Tristan\Documents\DATA\spotify_project\toptrack1.png', bbox_inches='tight')
plt.show()
```


    
![tracks1_png]( https://github.com/TristanT56/My-Spotify-Data-Analysis---Python/blob/main/Images%20for%20Readme%20markdown/output_106_0.png)
    



```python
#Top 10 track by nb of hours:
top10_track_h = spotify_tt.groupby(['track', 'artist'])['min_played'].sum().div(60).round(1).sort_values(ascending=False).reset_index().head(10)
top10_track_h.columns = ['track', 'artist', 'nb_of_hours']
top10_track_h
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>track</th>
      <th>artist</th>
      <th>nb_of_hours</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Here Comes That Sound Again</td>
      <td>Love De-Luxe</td>
      <td>2.6</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Sledgehammer</td>
      <td>Peter Gabriel</td>
      <td>2.5</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Nomalizo</td>
      <td>Letta Mbulu</td>
      <td>2.5</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Isabella</td>
      <td>Isaac Delusion</td>
      <td>2.3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Disco Inferno</td>
      <td>The Trammps</td>
      <td>2.2</td>
    </tr>
    <tr>
      <th>5</th>
      <td>I Feel Love</td>
      <td>Donna Summer</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Last Train to London</td>
      <td>Electric Light Orchestra</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Leader Of The Landslide</td>
      <td>The Lumineers</td>
      <td>1.9</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Esperar Pra Ver</td>
      <td>Poolside</td>
      <td>1.7</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Psycho Killer - 2005 Remaster</td>
      <td>Talking Heads</td>
      <td>1.6</td>
    </tr>
  </tbody>
</table>
</div>




```python
sns.set_style("white")
sns.barplot(x= 'track', y= 'nb_of_hours', data= top10_track_h, palette='winter_r')
plt.xticks(rotation=70,  ha="right", rotation_mode="anchor")
plt.xlabel(None)
plt.ylabel('Nb of hours\n')
plt.title('Top 10 tracks (Nb of hours)')

top10_track_h['H'] = top10_track_h.nb_of_hours.astype('str')
plt.legend(top10_track_h['track'] + ' - ' + top10_track_h['artist']+ '  ('+ top10_track_h['H'] +'H)',  loc = 2, bbox_to_anchor = (1,1))

plt.savefig(r'C:\Users\Tristan\Documents\DATA\spotify_project\toptrack2.png', bbox_inches='tight')
plt.show()
```


![tracks2_png]( https://github.com/TristanT56/My-Spotify-Data-Analysis---Python/blob/main/Images%20for%20Readme%20markdown/output_108_0.png)
    


### 3 - Top tracks of my top 1 artist: <a class="anchor" id="section_4_3"></a>


```python
#Top 10 track of The Lumineers by nb of plays:

Top1 = spotify_tt.loc[spotify_tt.artist.str.contains('Lumineers'), 'track'].value_counts().reset_index().head(10)
Top1.columns = ['track', 'nb_of_plays']
Top1
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>track</th>
      <th>nb_of_plays</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Gloria</td>
      <td>22</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Sleep On The Floor</td>
      <td>20</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Leader Of The Landslide</td>
      <td>20</td>
    </tr>
    <tr>
      <th>3</th>
      <td>It Wasn't Easy To Be Happy For You</td>
      <td>19</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Ophelia</td>
      <td>19</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Soundtrack Song - Bonus Track</td>
      <td>18</td>
    </tr>
    <tr>
      <th>6</th>
      <td>April</td>
      <td>17</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Cleopatra</td>
      <td>17</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Patience</td>
      <td>16</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Salt And The Sea</td>
      <td>16</td>
    </tr>
  </tbody>
</table>
</div>




```python
sns.set_style("white")
sns.barplot(x= 'track', y= 'nb_of_plays', data= Top1, palette='winter_r')
plt.xticks(rotation=70,  ha="right", rotation_mode="anchor")
plt.xlabel(None)
plt.ylabel('Nb of plays\n')
plt.title('Top 10 tracks (Nb of plays) of my Top 1 artist: The Lumineers')

Top1['P'] = Top1.nb_of_plays.astype('str')
plt.legend( Top1['track']+ '  ('+ Top1['P'] +')',  loc = 2, bbox_to_anchor = (1,1))

plt.savefig(r'C:\Users\Tristan\Documents\DATA\spotify_project\toptracktopartist.png', bbox_inches='tight')
plt.show()
```


![trackartist1_png]( https://github.com/TristanT56/My-Spotify-Data-Analysis---Python/blob/main/Images%20for%20Readme%20markdown/output_111_0.png)
    


### 4 - Creating a wordcloud visual with my top 100 artists: <a class="anchor" id="section_4_4"></a>


```python
#Let's create a wordcloud visual with my top 100 artists for the project image on my website
from wordcloud import WordCloud

wc_artist = spotify_tt['artist'].value_counts().head(100)
fig, ax = plt.subplots(figsize=(20,15))
wordcloud = WordCloud(width=1000,height=600, max_words=100,relative_scaling=0.78,normalize_plurals=False).generate_from_frequencies(wc_artist)
ax.imshow(wordcloud, interpolation='bilinear')

plt.savefig(r'C:\Users\Tristan\Documents\DATA\spotify_project\wc_top100.png', bbox_inches='tight')
plt.axis(False)
```




![wctop100_png]( https://github.com/TristanT56/My-Spotify-Data-Analysis---Python/blob/main/Images%20for%20Readme%20markdown/output_113_0.png)
    


### 5 - My music consumption on spotify per month during the analysis period (2020-07/2021-07): <a class="anchor" id="section_4_5"></a>


```python
month_h = spotify_tt.min_played.resample('M').sum().div(60).reset_index()
month_h.columns = ['month', 'nb_of_hours']
month_h
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>month</th>
      <th>nb_of_hours</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2020-07-31</td>
      <td>0.759305</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2020-08-31</td>
      <td>54.920501</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2020-09-30</td>
      <td>11.011556</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2020-10-31</td>
      <td>11.429484</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2020-11-30</td>
      <td>5.951911</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2020-12-31</td>
      <td>29.259204</td>
    </tr>
    <tr>
      <th>6</th>
      <td>2021-01-31</td>
      <td>12.622811</td>
    </tr>
    <tr>
      <th>7</th>
      <td>2021-02-28</td>
      <td>14.628442</td>
    </tr>
    <tr>
      <th>8</th>
      <td>2021-03-31</td>
      <td>19.739608</td>
    </tr>
    <tr>
      <th>9</th>
      <td>2021-04-30</td>
      <td>29.808107</td>
    </tr>
    <tr>
      <th>10</th>
      <td>2021-05-31</td>
      <td>23.025133</td>
    </tr>
    <tr>
      <th>11</th>
      <td>2021-06-30</td>
      <td>28.704854</td>
    </tr>
    <tr>
      <th>12</th>
      <td>2021-07-31</td>
      <td>43.618953</td>
    </tr>
  </tbody>
</table>
</div>




```python
from datetime import datetime

sns.set_theme(style="white")
fig, ax = plt.subplots()
fig = sns.lineplot(x= 'month', y = 'nb_of_hours' ,data = month_h, linewidth = 3)
sns.despine()
plt.xticks(month_h.month, rotation = 45, ha="right", rotation_mode="anchor")
plt.xlabel('\nMonths (2020-07/2021-07)')
plt.ylabel('Nb of hours\n')
ax.axvspan(datetime(2020,9,30), datetime(2020,12,15), alpha=0.1, color='red')
ax.axvspan(datetime(2021,4,3), datetime(2021,5,3), alpha=0.1, color='red')

ax.annotate('Lockdown\n in France',
            fontsize=10,
            fontweight='demi',
            xy=(datetime(2020,11,8), 45),  
            xycoords='data',
            xytext=(45, 10),      
            textcoords='offset points',
            arrowprops=dict(arrowstyle="->", color = 'black')) 
ax.annotate('',
            xy=(datetime(2021,4,25), 45),  
            xycoords='data',
            xytext=(-35, 10),      
            textcoords='offset points',
            arrowprops=dict(arrowstyle="->", color = 'black')) 

plt.title('My music consumption on spotify per month (From 2020-07 to 2021-07)')

plt.savefig(r'C:\Users\Tristan\Documents\DATA\spotify_project\months.png', bbox_inches='tight')
plt.show()
```


![months_png]( https://github.com/TristanT56/My-Spotify-Data-Analysis---Python/blob/main/Images%20for%20Readme%20markdown/output_116_0.png)
    


Interpretation: 
    
   We can see that I listen to spotify the most during the summer or during the Christmas and new eve period.
   We can also see that during the lockdown of 2020-09/2020-12 my music consumption on spotify dropped (It may not be a cause and effect relationship but it is a correlation). The 2021 lockdown does not seem to have affected my consumption.

### 6 - Which day I listen to spotify the most?: <a class="anchor" id="section_4_6"></a>


```python
day_h = spotify_tt.groupby(spotify_tt.index.date)['min_played'].sum().div(60).reset_index()
day_h.columns = ['weekday', 'nb_of_hours']
day_h['weekday'] = pd.to_datetime(day_h['weekday'])

weekday_h = day_h.groupby(day_h['weekday'].apply(lambda x: x.day_name()))['nb_of_hours'].mean().reset_index()
weekday_h['weekday'] = pd.Categorical(weekday_h['weekday'], ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
weekday_h = weekday_h.sort_values('weekday')
weekday_h
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>weekday</th>
      <th>nb_of_hours</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>Monday</td>
      <td>1.641194</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Tuesday</td>
      <td>1.540520</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Wednesday</td>
      <td>1.357016</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Thursday</td>
      <td>1.031079</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Friday</td>
      <td>1.559729</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Saturday</td>
      <td>2.104713</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Sunday</td>
      <td>1.293779</td>
    </tr>
  </tbody>
</table>
</div>




```python
sns.set_theme(style="whitegrid")
fig, ax = plt.subplots()
fig = sns.barplot(x= 'weekday', y = 'nb_of_hours' ,data = weekday_h, palette="Set2")
sns.despine()
plt.xticks(rotation = 45, ha="right", rotation_mode="anchor")
plt.xlabel(None)
plt.ylabel('Nb of hours (mean)\n')
plt.title('Which day I listen to spotify the most?')

plt.savefig(r'C:\Users\Tristan\Documents\DATA\spotify_project\days.png', bbox_inches='tight')
plt.show()
```


![days_png]( https://github.com/TristanT56/My-Spotify-Data-Analysis---Python/blob/main/Images%20for%20Readme%20markdown/output_120_0.png)
    


Interpretation: 
    
   Saturday seems to be my favourite day to listen to music on spotify and Thursday the day I listen to the least.

### 7 - Heatmap: When do I listen to Spotify the most during the week (by days and hours)?: <a class="anchor" id="section_4_7"></a>


```python
spotify_tt.reset_index(inplace=True)
spotify_tt['weekday'] = spotify_tt.datetime.apply(lambda x: x.day_name())
spotify_tt['hour'] = pd.DatetimeIndex(spotify_tt["datetime"]).hour
hmap_hourday = spotify_tt.groupby(['hour', 'weekday'])['min_played'].sum().div(60).reset_index()
hmap_hourday.columns = ['hour', 'weekday', 'nb_of_hours']
hmap_hourday['weekday'] = pd.Categorical(hmap_hourday['weekday'], ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
hmap_hourday = hmap_hourday.sort_values(['hour', 'weekday'])
hmap_hourday_pivot = hmap_hourday.pivot("hour", 'weekday', 'nb_of_hours')
hmap_hourday_pivot
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>weekday</th>
      <th>Monday</th>
      <th>Tuesday</th>
      <th>Wednesday</th>
      <th>Thursday</th>
      <th>Friday</th>
      <th>Saturday</th>
      <th>Sunday</th>
    </tr>
    <tr>
      <th>hour</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>NaN</td>
      <td>0.090550</td>
      <td>1.145496</td>
      <td>0.950365</td>
      <td>1.861331</td>
      <td>0.791024</td>
      <td>2.532518</td>
    </tr>
    <tr>
      <th>1</th>
      <td>NaN</td>
      <td>0.933529</td>
      <td>NaN</td>
      <td>0.060663</td>
      <td>1.664840</td>
      <td>0.206100</td>
      <td>1.244205</td>
    </tr>
    <tr>
      <th>2</th>
      <td>NaN</td>
      <td>0.013941</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.501499</td>
      <td>0.948349</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.922506</td>
      <td>0.434322</td>
      <td>0.997728</td>
      <td>0.187207</td>
    </tr>
    <tr>
      <th>4</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.986153</td>
      <td>0.209337</td>
      <td>0.592768</td>
      <td>0.962670</td>
    </tr>
    <tr>
      <th>5</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.128917</td>
      <td>1.419416</td>
      <td>1.054068</td>
      <td>0.913024</td>
    </tr>
    <tr>
      <th>6</th>
      <td>NaN</td>
      <td>0.848112</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.093284</td>
      <td>0.543313</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>7</th>
      <td>2.154869</td>
      <td>0.721891</td>
      <td>0.315204</td>
      <td>0.018783</td>
      <td>0.242081</td>
      <td>0.053570</td>
      <td>0.096764</td>
    </tr>
    <tr>
      <th>8</th>
      <td>1.769703</td>
      <td>0.820385</td>
      <td>1.963738</td>
      <td>0.815128</td>
      <td>2.254765</td>
      <td>2.400634</td>
      <td>0.180764</td>
    </tr>
    <tr>
      <th>9</th>
      <td>2.857846</td>
      <td>1.493154</td>
      <td>4.203145</td>
      <td>0.756811</td>
      <td>1.543869</td>
      <td>3.530228</td>
      <td>1.437077</td>
    </tr>
    <tr>
      <th>10</th>
      <td>2.070544</td>
      <td>1.371110</td>
      <td>3.375253</td>
      <td>1.078541</td>
      <td>3.498350</td>
      <td>3.374417</td>
      <td>2.896623</td>
    </tr>
    <tr>
      <th>11</th>
      <td>2.059680</td>
      <td>1.075490</td>
      <td>3.763706</td>
      <td>0.921599</td>
      <td>3.946406</td>
      <td>3.550700</td>
      <td>1.727983</td>
    </tr>
    <tr>
      <th>12</th>
      <td>2.460083</td>
      <td>2.109759</td>
      <td>4.860632</td>
      <td>2.268029</td>
      <td>6.334099</td>
      <td>1.219263</td>
      <td>2.150987</td>
    </tr>
    <tr>
      <th>13</th>
      <td>5.135331</td>
      <td>2.873739</td>
      <td>4.130917</td>
      <td>2.411043</td>
      <td>5.135137</td>
      <td>1.031386</td>
      <td>1.732020</td>
    </tr>
    <tr>
      <th>14</th>
      <td>4.815619</td>
      <td>4.349594</td>
      <td>3.419427</td>
      <td>2.264124</td>
      <td>4.091344</td>
      <td>2.703529</td>
      <td>2.331014</td>
    </tr>
    <tr>
      <th>15</th>
      <td>3.278965</td>
      <td>3.857129</td>
      <td>3.186157</td>
      <td>2.820599</td>
      <td>3.884292</td>
      <td>3.009397</td>
      <td>2.631171</td>
    </tr>
    <tr>
      <th>16</th>
      <td>2.042431</td>
      <td>2.919122</td>
      <td>2.517530</td>
      <td>2.515338</td>
      <td>1.224134</td>
      <td>2.016299</td>
      <td>3.550102</td>
    </tr>
    <tr>
      <th>17</th>
      <td>0.961435</td>
      <td>1.998544</td>
      <td>1.112515</td>
      <td>1.650704</td>
      <td>1.125251</td>
      <td>4.317831</td>
      <td>2.161906</td>
    </tr>
    <tr>
      <th>18</th>
      <td>2.253352</td>
      <td>1.074518</td>
      <td>1.174136</td>
      <td>0.745287</td>
      <td>1.297907</td>
      <td>5.773222</td>
      <td>1.517422</td>
    </tr>
    <tr>
      <th>19</th>
      <td>2.569555</td>
      <td>0.987227</td>
      <td>0.035959</td>
      <td>0.706920</td>
      <td>2.774501</td>
      <td>3.926154</td>
      <td>0.265591</td>
    </tr>
    <tr>
      <th>20</th>
      <td>1.416190</td>
      <td>0.165743</td>
      <td>NaN</td>
      <td>2.371770</td>
      <td>1.823672</td>
      <td>6.048907</td>
      <td>1.196161</td>
    </tr>
    <tr>
      <th>21</th>
      <td>0.956355</td>
      <td>0.674201</td>
      <td>0.319539</td>
      <td>3.366483</td>
      <td>3.476316</td>
      <td>4.371774</td>
      <td>0.559844</td>
    </tr>
    <tr>
      <th>22</th>
      <td>1.046890</td>
      <td>1.081222</td>
      <td>0.253854</td>
      <td>2.092066</td>
      <td>3.405139</td>
      <td>4.072307</td>
      <td>0.758668</td>
    </tr>
    <tr>
      <th>23</th>
      <td>1.539813</td>
      <td>1.351448</td>
      <td>0.862225</td>
      <td>2.111617</td>
      <td>2.349238</td>
      <td>4.503722</td>
      <td>0.016977</td>
    </tr>
  </tbody>
</table>
</div>




```python
fig, ax = plt.subplots(figsize=(3,6))
ax = sns.heatmap(hmap_hourday_pivot.fillna(0), robust=True, cmap="viridis",cbar_kws={'label': 'Sum of hours listenning music on Spotify'}, ax = ax);
ax.set(title="When do I listen to Spotify the most during the week? ", xlabel=None ,ylabel="Hour of the day")
colorbar = ax.collections[0].colorbar
colorbar.set_ticks([0,1,2,3,4,5])
colorbar.set_ticklabels(['0H', '1H', '2H', '3H', '4H', '5H'])

plt.savefig(r'C:\Users\Tristan\Documents\DATA\spotify_project\heatmap_weekday.png', bbox_inches='tight')
plt.show()
```


![hours_png]( https://github.com/TristanT56/My-Spotify-Data-Analysis---Python/blob/main/Images%20for%20Readme%20markdown/output_124_0.png)
    


Interpretation: 

We can see that between Monday and Friday, I listen the most between 9H and 15H (my peak hours are during my lunch break: 12H, 13H and 14H). During the weekend, I listen the most the Saturday afternoon after 17H.

## D) Analysis Part 2 - Audio features<a class="anchor" id="chapter5"></a>

In this second part of the analysis, my objectives are:
- Find the tracks corresponding to the max and min for each audio feature.
- Find the audio features corresponding to all the songs I listened to. 
- Compare it to the audio features of my top tracks.
- Analyse the audio features corresponding to the tracks I listen to in the morning, afternoon, evening and night/party.
- Using these findings and only the audio features, find a track I might like to listen to in the morning/afternoon/evening/and during a party at night.

For this part, we will use the full table with the 'spotify_features'.

### 1 - The tracks corresponding to the max and min for each audio feature: <a class="anchor" id="section_5_1"></a>


```python
d = spotify_features.loc[spotify_features.danceability == spotify_features.danceability.max()].drop_duplicates(subset = ['track','artist'])
e = spotify_features.loc[spotify_features.energy == spotify_features.energy.max()].drop_duplicates(subset = ['track','artist'])
v = spotify_features.loc[spotify_features.valence == spotify_features.valence.max()].drop_duplicates(subset = ['track','artist'])
l = spotify_features.loc[spotify_features.loudness == spotify_features.loudness.max()].drop_duplicates(subset = ['track','artist'])
i = spotify_features.loc[spotify_features.instrumentalness == spotify_features.instrumentalness.max()].drop_duplicates(subset = ['track','artist'])
a = spotify_features.loc[spotify_features.acousticness == spotify_features.acousticness.max()].drop_duplicates(subset = ['track','artist'])
dd = spotify_features.loc[spotify_features.danceability == spotify_features.danceability.min()].drop_duplicates(subset = ['track','artist'])
ee = spotify_features.loc[spotify_features.energy == spotify_features.energy.min()].drop_duplicates(subset = ['track','artist'])
vv = spotify_features.loc[spotify_features.valence == spotify_features.valence.min()].drop_duplicates(subset = ['track','artist'])
ll = spotify_features.loc[spotify_features.loudness == spotify_features.loudness.min()].drop_duplicates(subset = ['track','artist'])
ii = spotify_features.loc[spotify_features.instrumentalness == spotify_features.instrumentalness.min()].drop_duplicates(subset = ['track','artist'])
aa = spotify_features.loc[spotify_features.acousticness == spotify_features.acousticness.min()].drop_duplicates(subset = ['track','artist'])

print('\n----MAX danceability')
print(d[['track', 'artist']])
print('\n----MIN danceability')
print(dd[['track', 'artist']])
print('\n ')
print('\n----MAX energy')
print(e[['track', 'artist']])
print('\n----MIN energy')
print(ee[['track', 'artist']])
print('\n ')
print('\n----MAX valence')
print(v[['track', 'artist']])
print('\n----MIN valence')
print(vv[['track', 'artist']])
print('\n ')
print('\n----MAX loudness')
print(l[['track', 'artist']])
print('\n----MIN loudness')
print(ll[['track', 'artist']])
print('\n ')
print('\n----MAX instrumentalness')
print(i[['track', 'artist']])
print('\n----MIN instrumentalness')
print(ii[['track', 'artist']].sample(2, random_state=10))
print('\n ')
print('\n----MAX acousticness')
print(a[['track', 'artist']])
print('\n----MIN acousticness')
print(aa[['track', 'artist']])
```

    
    ----MAX danceability
                                track   artist
    2760  Credits (feat. Dean Dawson)  Parcels
    
    ----MIN danceability
                      track           artist
    3531  L'estasi dell'oro  Ennio Morricone
    
     
    
    ----MAX energy
                        track     artist
    2974    Hippy Hippy Shake   Big Soul
    4200  TURN OFF THE LIGHTS  Dog Blood
    
    ----MIN energy
                   track          artist
    3635  Claire de lune  Claude Debussy
    
     
    
    ----MAX valence
                 track    artist
    384  Gotta Go Home  Boney M.
    
    ----MIN valence
          track artist
    3172  Ephos   Flug
    
     
    
    ----MAX loudness
              track                    artist
    205  Menez daou  Les Ramoneurs De Menhirs
    
    ----MIN loudness
                   track          artist
    3635  Claire de lune  Claude Debussy
    
     
    
    ----MAX instrumentalness
                      track      artist
    4639  Bilboquet (Sirba)  Polo & Pan
    
    ----MIN instrumentalness
                track      artist
    1004  Gold Digger  Kanye West
    2047    La source        1995
    
     
    
    ----MAX acousticness
                        track          artist
    3056  This Way Or Another    Owen Kennedy
    3635       Claire de lune  Claude Debussy
    
    ----MIN acousticness
                   track     artist
    4569  Good Samaritan  The Hives
    

### 2 - The audio features corresponding to all the tracks I listened to: <a class="anchor" id="section_5_2"></a>


```python
spotify_features.iloc[:,4:10].mean().reset_index()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>index</th>
      <th>0</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>danceability</td>
      <td>0.646954</td>
    </tr>
    <tr>
      <th>1</th>
      <td>energy</td>
      <td>0.634980</td>
    </tr>
    <tr>
      <th>2</th>
      <td>valence</td>
      <td>0.572766</td>
    </tr>
    <tr>
      <th>3</th>
      <td>loudness</td>
      <td>0.678102</td>
    </tr>
    <tr>
      <th>4</th>
      <td>instrumentalness</td>
      <td>0.173087</td>
    </tr>
    <tr>
      <th>5</th>
      <td>acousticness</td>
      <td>0.264564</td>
    </tr>
  </tbody>
</table>
</div>




```python
import plotly.graph_objects as go

categories = ['Danceability', 'Energy', 'Valence', 'Loudness','Instrumentalness', 'Acousticness']

fig = go.Figure()

fig.add_trace(go.Scatterpolar(
      r= spotify_features.iloc[:,4:10].mean(),
      theta=categories,
      fill='toself',
      name = 'All tracks in my Spotify history (2020/2021): average weighted by nb of plays.'
))

fig.update_layout(
    title = "Audio features corresponding to all the tracks I listened to",
    polar=dict(
    radialaxis=dict(
      visible=True,
      range=[0, 1]
    )),
  showlegend=True)

fig.update_layout(legend=dict(
    orientation="h",
    yanchor="bottom",
    y=-0.3,
    xanchor="left",
    x=0
))

fig.write_image(r'C:\Users\Tristan\Documents\DATA\spotify_project\all_tracks.png')
fig.show()
```

![alltracks_png]( https://github.com/TristanT56/My-Spotify-Data-Analysis---Python/blob/main/Images%20for%20Readme%20markdown/output_132_0.png)

Interpretation: 
    
Although my tastes are diverse ([see Exploratory Data Analysis results in part B](#eda)), I seem to prefer music that is quite danceable with some energy and also quite loud. I prefer music with a good valence. I prefer music that is not acoustic and that has vocals.

### 3 - Audio features: all tracks VS  top tracks: <a class="anchor" id="section_5_3"></a>


```python
#My top 5 tracks
top5 = spotify_features[['track', 'artist']].value_counts().reset_index().head(5)
top5.columns = ['track','artist' ,'nb_of_plays']
top5
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>track</th>
      <th>artist</th>
      <th>nb_of_plays</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Disco Inferno</td>
      <td>The Trammps</td>
      <td>40</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Sledgehammer</td>
      <td>Peter Gabriel</td>
      <td>33</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Nomalizo</td>
      <td>Letta Mbulu</td>
      <td>32</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Nothing But A Heartache</td>
      <td>The Flirtations</td>
      <td>31</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Last Train to London</td>
      <td>Electric Light Orchestra</td>
      <td>30</td>
    </tr>
  </tbody>
</table>
</div>




```python
#My top 1 track from my top 1 artist
top1_artist = spotify_features.loc[spotify_features.artist.str.contains('Lumineers'), ['track', 'artist']].value_counts().reset_index().head(1)
top1_artist.columns = ['track','artist' ,'nb_of_plays']
top1_artist
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>track</th>
      <th>artist</th>
      <th>nb_of_plays</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Gloria</td>
      <td>The Lumineers</td>
      <td>22</td>
    </tr>
  </tbody>
</table>
</div>




```python
top5_top1artist = pd.concat([top5, top1_artist])
top5_top1artist
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>track</th>
      <th>artist</th>
      <th>nb_of_plays</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Disco Inferno</td>
      <td>The Trammps</td>
      <td>40</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Sledgehammer</td>
      <td>Peter Gabriel</td>
      <td>33</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Nomalizo</td>
      <td>Letta Mbulu</td>
      <td>32</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Nothing But A Heartache</td>
      <td>The Flirtations</td>
      <td>31</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Last Train to London</td>
      <td>Electric Light Orchestra</td>
      <td>30</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Gloria</td>
      <td>The Lumineers</td>
      <td>22</td>
    </tr>
  </tbody>
</table>
</div>




```python
top5_top1artist_features = spotify_features.loc[(spotify_features['track'].isin(top5_top1artist['track'])) &
                           (spotify_features['artist'].isin(top5_top1artist['artist']))]
top5_top1artist_features.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>datetime</th>
      <th>track</th>
      <th>artist</th>
      <th>min_played</th>
      <th>danceability</th>
      <th>energy</th>
      <th>valence</th>
      <th>loudness</th>
      <th>instrumentalness</th>
      <th>acousticness</th>
      <th>tempo</th>
      <th>mode</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>247</th>
      <td>2020-08-12 10:31:00</td>
      <td>Last Train to London</td>
      <td>Electric Light Orchestra</td>
      <td>4.315017</td>
      <td>0.702735</td>
      <td>0.535569</td>
      <td>0.972643</td>
      <td>0.610962</td>
      <td>0.00082</td>
      <td>0.397989</td>
      <td>121.493</td>
      <td>1</td>
    </tr>
    <tr>
      <th>248</th>
      <td>2020-08-15 20:16:00</td>
      <td>Last Train to London</td>
      <td>Electric Light Orchestra</td>
      <td>4.499267</td>
      <td>0.702735</td>
      <td>0.535569</td>
      <td>0.972643</td>
      <td>0.610962</td>
      <td>0.00082</td>
      <td>0.397989</td>
      <td>121.493</td>
      <td>1</td>
    </tr>
    <tr>
      <th>249</th>
      <td>2020-08-17 17:57:00</td>
      <td>Last Train to London</td>
      <td>Electric Light Orchestra</td>
      <td>4.499533</td>
      <td>0.702735</td>
      <td>0.535569</td>
      <td>0.972643</td>
      <td>0.610962</td>
      <td>0.00082</td>
      <td>0.397989</td>
      <td>121.493</td>
      <td>1</td>
    </tr>
    <tr>
      <th>250</th>
      <td>2020-08-25 15:56:00</td>
      <td>Last Train to London</td>
      <td>Electric Light Orchestra</td>
      <td>4.490467</td>
      <td>0.702735</td>
      <td>0.535569</td>
      <td>0.972643</td>
      <td>0.610962</td>
      <td>0.00082</td>
      <td>0.397989</td>
      <td>121.493</td>
      <td>1</td>
    </tr>
    <tr>
      <th>251</th>
      <td>2020-08-26 16:30:00</td>
      <td>Last Train to London</td>
      <td>Electric Light Orchestra</td>
      <td>4.499100</td>
      <td>0.702735</td>
      <td>0.535569</td>
      <td>0.972643</td>
      <td>0.610962</td>
      <td>0.00082</td>
      <td>0.397989</td>
      <td>121.493</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
top_tracks = top5_top1artist_features.iloc[:, 4:10].mean().reset_index()
all_tracks = spotify_features.iloc[:,4:10].mean().reset_index()
vs = top_tracks.merge(all_tracks, on= 'index')
vs.rename(columns = {'index':'audio_feature',   '0_x':'top_tracks', '0_y':'all_tracks'}, inplace =True)
vs['diff'] =  vs['top_tracks'] - vs['all_tracks']
vs
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>audio_feature</th>
      <th>top_tracks</th>
      <th>all_tracks</th>
      <th>diff</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>danceability</td>
      <td>0.646419</td>
      <td>0.646954</td>
      <td>-0.000535</td>
    </tr>
    <tr>
      <th>1</th>
      <td>energy</td>
      <td>0.706922</td>
      <td>0.634980</td>
      <td>0.071942</td>
    </tr>
    <tr>
      <th>2</th>
      <td>valence</td>
      <td>0.746982</td>
      <td>0.572766</td>
      <td>0.174216</td>
    </tr>
    <tr>
      <th>3</th>
      <td>loudness</td>
      <td>0.731358</td>
      <td>0.678102</td>
      <td>0.053256</td>
    </tr>
    <tr>
      <th>4</th>
      <td>instrumentalness</td>
      <td>0.134393</td>
      <td>0.173087</td>
      <td>-0.038694</td>
    </tr>
    <tr>
      <th>5</th>
      <td>acousticness</td>
      <td>0.223034</td>
      <td>0.264564</td>
      <td>-0.041530</td>
    </tr>
  </tbody>
</table>
</div>




```python
from plotly.subplots import make_subplots

pyo.init_notebook_mode()

categories = ['Danceability', 'Energy', 'Valence', 'Loudness','Instrumentalness', 'Acousticness']



fig = make_subplots(rows=1, cols=1, specs=[[{'type': 'polar'}]])


fig.add_trace(go.Scatterpolar(
      r= top5_top1artist_features.iloc[:, 4:10].mean(),
      theta=categories,
      fill='toself',
      name='Top 5 tracks & top 1 track of my top artist: average weighted by nb of plays.'
), 1,1)


fig.add_trace(go.Scatterpolar(
      r= spotify_features.iloc[:,4:10].mean(),
      theta=categories,
      fill='toself',
      name = 'All tracks in my Spotify history (2020/2021): average weighted by nb of plays.'
), 1,1)



fig.update_layout(
    title = "Audio features: all tracks VS top tracks\n",
    
    polar=dict(
    angularaxis_showticklabels=True,
    radialaxis_showticklabels=True,
    radialaxis=dict(
     visible=True,
      range=[0, 1])),
  showlegend=True)


fig.update_layout(legend=dict(
    orientation="h",
    yanchor="bottom",
    y=-0.3,
    xanchor="right",
    x=1
))


fig.write_image(r'C:\Users\Tristan\Documents\DATA\spotify_project\all_vs_tops.png')
pyo.iplot(fig, filename = 'all_vs_tops_tracks')
```

![allvstoptracks_png]( https://github.com/TristanT56/My-Spotify-Data-Analysis---Python/blob/main/Images%20for%20Readme%20markdown/output_141_0.png)
