
# Analysis of Goolge Merchant Store Analytics data

Google Analytics had conducted a competition on Kaggle that aimed to predict revenue users generate based on their behviour captured by Google Analytics themselves. Though the competition suffered from feature leakage and the fact that participants found the complete dataset online; when using the data provided , several key concepts of data science and ML can be put to use here.

# Problem statement and nature of data
Google wanted to predict transaction a particular visitor might generate when they visit the google merchant website. A dataset was provided in form of a flatfile with embedded JSON which was to be used to find the best trends.

### Proceeding with imports to provide a view of the data


```python
import json
from pprint import pprint
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import numpy as np
import datetime as dt
import urllib
import  pickle
import sys, os
import re
pd.set_option('display.max_rows', 5000)
pd.set_option('display.max_columns', 5000)
pd.set_option('display.width', 10000)
pd.options.display.max_colwidth = 1000
## Above options to provide a decent view of the data in tablular form
```


```python
#trainRaw = pd.read_csv(r'C:\Users\This PC\Documents\NewTechnologies\MachineLearning\data\GoogleAnalytics\all\train.csv' , dtype = {'fullVisitorId':'|S'})
## Column fullVisitorId is used as String as it has leading zeros in some instances
```


```python
trainRaw.head(1)
```

The first row from the flat file shows the arrnagement of the JSON structure viz. Stored as columns


```python

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
      <th>channelGrouping</th>
      <th>date</th>
      <th>device</th>
      <th>fullVisitorId</th>
      <th>geoNetwork</th>
      <th>sessionId</th>
      <th>socialEngagementType</th>
      <th>totals</th>
      <th>trafficSource</th>
      <th>visitId</th>
      <th>visitNumber</th>
      <th>visitStartTime</th>
      <th>visitStartTime_x</th>
      <th>diff</th>
      <th>visits</th>
      <th>hits</th>
      <th>pageviews</th>
      <th>bototalsunces</th>
      <th>newVisits</th>
      <th>transactionRevenue</th>
      <th>browser</th>
      <th>browserVersion</th>
      <th>browserSize</th>
      <th>operatingSystem</th>
      <th>operatingSystemVersion</th>
      <th>isMobile</th>
      <th>mobileDeviceBranding</th>
      <th>mobileDeviceModel</th>
      <th>mobileInputSelector</th>
      <th>mobileDeviceInfo</th>
      <th>mobileDeviceMarketingName</th>
      <th>flashVersion</th>
      <th>language</th>
      <th>screenColors</th>
      <th>screenResolution</th>
      <th>deviceCategory</th>
      <th>campaign</th>
      <th>source</th>
      <th>medium</th>
      <th>keyword</th>
      <th>isTrueDirect</th>
      <th>referralPath</th>
      <th>adContent</th>
      <th>campaignCode</th>
      <th>continent</th>
      <th>subContinent</th>
      <th>country</th>
      <th>region</th>
      <th>metro</th>
      <th>city</th>
      <th>cityId</th>
      <th>networkDomain</th>
      <th>latitude</th>
      <th>longitude</th>
      <th>networkLocation</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Organic Search</td>
      <td>20160902</td>
      <td>{"browser": "Chrome", "browserVersion": "not available in demo dataset", "browserSize": "not available in demo dataset", "operatingSystem": "Windows", "operatingSystemVersion": "not available in demo dataset", "isMobile": false, "mobileDeviceBranding": "not available in demo dataset", "mobileDeviceModel": "not available in demo dataset", "mobileInputSelector": "not available in demo dataset", "mobileDeviceInfo": "not available in demo dataset", "mobileDeviceMarketingName": "not available in demo dataset", "flashVersion": "not available in demo dataset", "language": "not available in demo dataset", "screenColors": "not available in demo dataset", "screenResolution": "not available in demo dataset", "deviceCategory": "desktop"}</td>
      <td>1131660440785968503</td>
      <td>{"continent": "Asia", "subContinent": "Western Asia", "country": "Turkey", "region": "Izmir", "metro": "(not set)", "city": "Izmir", "cityId": "not available in demo dataset", "networkDomain": "ttnet.com.tr", "latitude": "not available in demo dataset", "longitude": "not available in demo dataset", "networkLocation": "not available in demo dataset"}</td>
      <td>1131660440785968503_1472830385</td>
      <td>Not Socially Engaged</td>
      <td>{"visits": "1", "hits": "1", "pageviews": "1", "bounces": "1", "newVisits": "1"}</td>
      <td>{"campaign": "(not set)", "source": "google", "medium": "organic", "keyword": "(not provided)", "adwordsClickInfo": {"criteriaParameters": "not available in demo dataset"}}</td>
      <td>1472830385</td>
      <td>1</td>
      <td>1472830385</td>
      <td>1472830385</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>Chrome</td>
      <td>not available in demo dataset</td>
      <td>not available in demo dataset</td>
      <td>Windows</td>
      <td>not available in demo dataset</td>
      <td>False</td>
      <td>not available in demo dataset</td>
      <td>not available in demo dataset</td>
      <td>not available in demo dataset</td>
      <td>not available in demo dataset</td>
      <td>not available in demo dataset</td>
      <td>not available in demo dataset</td>
      <td>not available in demo dataset</td>
      <td>not available in demo dataset</td>
      <td>not available in demo dataset</td>
      <td>desktop</td>
      <td>(not set)</td>
      <td>google</td>
      <td>organic</td>
      <td>(not provided)</td>
      <td>yy</td>
      <td>yy</td>
      <td>yy</td>
      <td>yy</td>
      <td>Asia</td>
      <td>Western Asia</td>
      <td>Turkey</td>
      <td>Izmir</td>
      <td>(not set)</td>
      <td>Izmir</td>
      <td>not available in demo dataset</td>
      <td>ttnet.com.tr</td>
      <td>not available in demo dataset</td>
      <td>not available in demo dataset</td>
      <td>not available in demo dataset</td>
    </tr>
  </tbody>
</table>
</div>



The data was converted to a flat strucutre using a JSON parsing code : Courtesy one of the participants who shared
# This script read the csvs, delete useless columns, set datatypes
# and save dataframes as pickle files for faster loading

import os
import numpy as np
import pandas as pd
from pandas.io.json import json_normalize
import json
import time

pd.options.display.max_columns = 999

def load_df(file_name, nrows=None):
    """ Read csv and convert json columns. Author: Julián Peller. """
    
    JSON_COLUMNS = ['device', 'geoNetwork', 'totals', 'trafficSource']
    df = pd.read_csv(file_name,
                     converters={column: json.loads for column in JSON_COLUMNS}, 
                     dtype={'fullVisitorId': 'str'}, nrows=nrows)
    
    for column in JSON_COLUMNS:
        column_as_df = json_normalize(df[column])
        column_as_df.columns = [f"{column}_{subcolumn}" for subcolumn in column_as_df.columns]
        df = df.drop(column, axis=1).merge(column_as_df, right_index=True, left_index=True)
    return df

    # Read csv
timer = time.time()
#train = load_df(r'C:\Users\This PC\Documents\NewTechnologies\MachineLearning\data\GoogleAnalytics\all\train.csv' )
#test = load_df(r'C:\Users\This PC\Documents\NewTechnologies\MachineLearning\data\GoogleAnalytics\all\test.csv')

The data that was previously pickled is loaded into pandas DF again
with open(r'C:\Users\This PC\Documents\NewTechnologies\MachineLearning\data\GoogleAnalytics\all\pickles\train_clean1.pkl','rb') as filePointer:
    train = pickle.load(filePointer)



```python
train.head(1)
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
      <th>channelGrouping</th>
      <th>date</th>
      <th>fullVisitorId</th>
      <th>sessionId</th>
      <th>socialEngagementType</th>
      <th>visitId</th>
      <th>visitNumber</th>
      <th>visitStartTime</th>
      <th>device_browser</th>
      <th>device_browserSize</th>
      <th>device_browserVersion</th>
      <th>device_deviceCategory</th>
      <th>device_flashVersion</th>
      <th>device_isMobile</th>
      <th>device_language</th>
      <th>device_mobileDeviceBranding</th>
      <th>device_mobileDeviceInfo</th>
      <th>device_mobileDeviceMarketingName</th>
      <th>device_mobileDeviceModel</th>
      <th>device_mobileInputSelector</th>
      <th>device_operatingSystem</th>
      <th>device_operatingSystemVersion</th>
      <th>device_screenColors</th>
      <th>device_screenResolution</th>
      <th>geoNetwork_city</th>
      <th>geoNetwork_cityId</th>
      <th>geoNetwork_continent</th>
      <th>geoNetwork_country</th>
      <th>geoNetwork_latitude</th>
      <th>geoNetwork_longitude</th>
      <th>geoNetwork_metro</th>
      <th>geoNetwork_networkDomain</th>
      <th>geoNetwork_networkLocation</th>
      <th>geoNetwork_region</th>
      <th>geoNetwork_subContinent</th>
      <th>totals_bounces</th>
      <th>totals_hits</th>
      <th>totals_newVisits</th>
      <th>totals_pageviews</th>
      <th>totals_transactionRevenue</th>
      <th>totals_visits</th>
      <th>trafficSource_adContent</th>
      <th>trafficSource_adwordsClickInfo.adNetworkType</th>
      <th>trafficSource_adwordsClickInfo.criteriaParameters</th>
      <th>trafficSource_adwordsClickInfo.gclId</th>
      <th>trafficSource_adwordsClickInfo.isVideoAd</th>
      <th>trafficSource_adwordsClickInfo.page</th>
      <th>trafficSource_adwordsClickInfo.slot</th>
      <th>trafficSource_campaign</th>
      <th>trafficSource_campaignCode</th>
      <th>trafficSource_isTrueDirect</th>
      <th>trafficSource_keyword</th>
      <th>trafficSource_medium</th>
      <th>trafficSource_referralPath</th>
      <th>trafficSource_source</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Organic Search</td>
      <td>20160902</td>
      <td>1131660440785968503</td>
      <td>1131660440785968503_1472830385</td>
      <td>Not Socially Engaged</td>
      <td>1472830385</td>
      <td>1</td>
      <td>1472830385</td>
      <td>Chrome</td>
      <td>not available in demo dataset</td>
      <td>not available in demo dataset</td>
      <td>desktop</td>
      <td>not available in demo dataset</td>
      <td>False</td>
      <td>not available in demo dataset</td>
      <td>not available in demo dataset</td>
      <td>not available in demo dataset</td>
      <td>not available in demo dataset</td>
      <td>not available in demo dataset</td>
      <td>not available in demo dataset</td>
      <td>Windows</td>
      <td>not available in demo dataset</td>
      <td>not available in demo dataset</td>
      <td>not available in demo dataset</td>
      <td>Izmir</td>
      <td>not available in demo dataset</td>
      <td>Asia</td>
      <td>Turkey</td>
      <td>not available in demo dataset</td>
      <td>not available in demo dataset</td>
      <td>(not set)</td>
      <td>ttnet.com.tr</td>
      <td>not available in demo dataset</td>
      <td>Izmir</td>
      <td>Western Asia</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>NaN</td>
      <td>1</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>not available in demo dataset</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>(not set)</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>(not provided)</td>
      <td>organic</td>
      <td>NaN</td>
      <td>google</td>
    </tr>
  </tbody>
</table>
</div>



# Important feature discovery
It was observed that apart from the usual features , the timezone of the request was extremely helpful.Argument being, many individuals would prefer to shop during a convinient time in the day and it might lead to a purchase.
The time in the dataset was the time of the IP destination and not the time of the origin city . 

### We hence took some commercially available geolocation and timezone apis and created a file that contained the timezone and time offset of the cities that have appeared in the data. This timezone info was offset to get the local time when the request when the visit was made. Moreover using cities we can also get an idea of the probably currency of transaction and adjust the transaction amounts whose units are kept hidden


```python
locations = pd.read_csv(r'C:\Users\This PC\Documents\NewTechnologies\MachineLearning\data\GoogleAnalytics\all\export\cityrequestfile.txt')
locations.head(2)
import requests
file = open(r'C:\Users\This PC\Documents\NewTechnologies\MachineLearning\data\GoogleAnalytics\all\export\latlong.csv','a') 
for locnames in locations.location:
    r = requests.get(r'https://nominatim.openstreetmap.org/search?q='+locnames+'&format=json&addressdetails=0')
    t=json.loads(r.content)
    for i in range(1):
        print(locnames+','+t[i]['lat']+','+t[i]['lon'])
        file.writelines(locnames+','+t[i]['lat']+','+t[i]['lon']+'\n')
file.close()

```

### Shwon below is a sample of the responses recived from the request. We used google geolocation and timezone apis

#### Examples
Tajikistan,38.6281733,70.8156541<br>
Tanzania,-6.5247123,35.7878438<br>
Thailand,14.8971921,100.83273<br>
Hua Hin,12.5678487,99.9575368<br>
Nakhon Pathom,13.8918425,100.0165659<br>
Timor-Leste,-8.5151979,125.8375756<br>
Togo,8.7800265,1.0199765<br>
Trinidad & Tobago,37.169397,-104.5005411<br>
Tunisia,33.8439408,9.400138<br>
Turkey,38.9597594,34.9249653<br>




```python
locationtimezoneoffsets = pd.read_csv(r'C:\Users\This PC\Documents\NewTechnologies\MachineLearning\data\GoogleAnalytics\all\export\resp1.txt' , sep='~')
locationtimezoneoffsets = locationtimezoneoffsets.rename({'location':'geoNetwork_city'} , axis =1)
locationtimezoneoffsets.head(1)

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
      <th>geoNetwork_city</th>
      <th>dst</th>
      <th>timezoneId</th>
      <th>offset</th>
      <th>timezoneName</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Manila</td>
      <td>0</td>
      <td>Asia/Manila</td>
      <td>28800</td>
      <td>Philippine Standard Time</td>
    </tr>
  </tbody>
</table>
</div>



The next steps being performed are to clean some remaining data. , Default values that do not exist and impute some non numerical values into numeric features
Most of the code is self explanatory. The fetures being used finally will be described in the sections further below



```python
## Renaming a column before the Join
train2=train.copy()
train = train.merge(locationtimezoneoffsets , left_on =['geoNetwork_city'] ,right_on=['geoNetwork_city'] , how = 'left')
locationtimezoneoffsets = locationtimezoneoffsets.rename({'geoNetwork_city':'geoNetwork_country'} , axis =1) 
#Join is peformed between timezone file and the train data
train2 = train2.merge(locationtimezoneoffsets , left_on =['geoNetwork_country'] ,right_on=['geoNetwork_country'] , how = 'left')

```


```python
train.head(1)
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
      <th>channelGrouping</th>
      <th>date</th>
      <th>fullVisitorId</th>
      <th>sessionId</th>
      <th>socialEngagementType</th>
      <th>visitId</th>
      <th>visitNumber</th>
      <th>visitStartTime</th>
      <th>device_browser</th>
      <th>device_browserSize</th>
      <th>device_browserVersion</th>
      <th>device_deviceCategory</th>
      <th>device_flashVersion</th>
      <th>device_isMobile</th>
      <th>device_language</th>
      <th>device_mobileDeviceBranding</th>
      <th>device_mobileDeviceInfo</th>
      <th>device_mobileDeviceMarketingName</th>
      <th>device_mobileDeviceModel</th>
      <th>device_mobileInputSelector</th>
      <th>device_operatingSystem</th>
      <th>device_operatingSystemVersion</th>
      <th>device_screenColors</th>
      <th>device_screenResolution</th>
      <th>geoNetwork_city</th>
      <th>geoNetwork_cityId</th>
      <th>geoNetwork_continent</th>
      <th>geoNetwork_country</th>
      <th>geoNetwork_latitude</th>
      <th>geoNetwork_longitude</th>
      <th>geoNetwork_metro</th>
      <th>geoNetwork_networkDomain</th>
      <th>geoNetwork_networkLocation</th>
      <th>geoNetwork_region</th>
      <th>geoNetwork_subContinent</th>
      <th>totals_bounces</th>
      <th>totals_hits</th>
      <th>totals_newVisits</th>
      <th>totals_pageviews</th>
      <th>totals_transactionRevenue</th>
      <th>totals_visits</th>
      <th>trafficSource_adContent</th>
      <th>trafficSource_adwordsClickInfo.adNetworkType</th>
      <th>trafficSource_adwordsClickInfo.criteriaParameters</th>
      <th>trafficSource_adwordsClickInfo.gclId</th>
      <th>trafficSource_adwordsClickInfo.isVideoAd</th>
      <th>trafficSource_adwordsClickInfo.page</th>
      <th>trafficSource_adwordsClickInfo.slot</th>
      <th>trafficSource_campaign</th>
      <th>trafficSource_campaignCode</th>
      <th>trafficSource_isTrueDirect</th>
      <th>trafficSource_keyword</th>
      <th>trafficSource_medium</th>
      <th>trafficSource_referralPath</th>
      <th>trafficSource_source</th>
      <th>dst</th>
      <th>timezoneId</th>
      <th>offset</th>
      <th>timezoneName</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Organic Search</td>
      <td>20160902</td>
      <td>1131660440785968503</td>
      <td>1131660440785968503_1472830385</td>
      <td>Not Socially Engaged</td>
      <td>1472830385</td>
      <td>1</td>
      <td>1472830385</td>
      <td>Chrome</td>
      <td>not available in demo dataset</td>
      <td>not available in demo dataset</td>
      <td>desktop</td>
      <td>not available in demo dataset</td>
      <td>False</td>
      <td>not available in demo dataset</td>
      <td>not available in demo dataset</td>
      <td>not available in demo dataset</td>
      <td>not available in demo dataset</td>
      <td>not available in demo dataset</td>
      <td>not available in demo dataset</td>
      <td>Windows</td>
      <td>not available in demo dataset</td>
      <td>not available in demo dataset</td>
      <td>not available in demo dataset</td>
      <td>Izmir</td>
      <td>not available in demo dataset</td>
      <td>Asia</td>
      <td>Turkey</td>
      <td>not available in demo dataset</td>
      <td>not available in demo dataset</td>
      <td>(not set)</td>
      <td>ttnet.com.tr</td>
      <td>not available in demo dataset</td>
      <td>Izmir</td>
      <td>Western Asia</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>NaN</td>
      <td>1</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>not available in demo dataset</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>(not set)</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>(not provided)</td>
      <td>organic</td>
      <td>NaN</td>
      <td>google</td>
      <td>0.0</td>
      <td>Europe/Istanbul</td>
      <td>7200.0</td>
      <td>Eastern European Standard Time</td>
    </tr>
  </tbody>
</table>
</div>




```python
train_city = train.loc[~train.offset.isnull()]
train_ctry = train2.loc[~(train2.offset.isnull()) & ~(train2.sessionId.isin(train_city.sessionId))]
train_tz = train_ctry.append(train_city, ignore_index=True)
train_tz = train_tz.assign(unixtimeinzone = lambda x : x['offset'].astype(np.int64) + x['visitStartTime'].astype(np.int64) )
train_tz = train_tz.assign (month =  lambda x : pd.to_datetime(x['unixtimeinzone'], unit='s').dt.month, hour =  lambda x : pd.to_datetime(x['unixtimeinzone'], unit='s').dt.hour  ,day =  lambda x : pd.to_datetime(x['unixtimeinzone'], unit='s').dt.day )
## The offsets are added to the local time is derived

```


```python
train_tz.head(2)
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
      <th>channelGrouping</th>
      <th>date</th>
      <th>fullVisitorId</th>
      <th>sessionId</th>
      <th>socialEngagementType</th>
      <th>visitId</th>
      <th>visitNumber</th>
      <th>visitStartTime</th>
      <th>device_browser</th>
      <th>device_browserSize</th>
      <th>device_browserVersion</th>
      <th>device_deviceCategory</th>
      <th>device_flashVersion</th>
      <th>device_isMobile</th>
      <th>device_language</th>
      <th>device_mobileDeviceBranding</th>
      <th>device_mobileDeviceInfo</th>
      <th>device_mobileDeviceMarketingName</th>
      <th>device_mobileDeviceModel</th>
      <th>device_mobileInputSelector</th>
      <th>device_operatingSystem</th>
      <th>device_operatingSystemVersion</th>
      <th>device_screenColors</th>
      <th>device_screenResolution</th>
      <th>geoNetwork_city</th>
      <th>geoNetwork_cityId</th>
      <th>geoNetwork_continent</th>
      <th>geoNetwork_country</th>
      <th>geoNetwork_latitude</th>
      <th>geoNetwork_longitude</th>
      <th>geoNetwork_metro</th>
      <th>geoNetwork_networkDomain</th>
      <th>geoNetwork_networkLocation</th>
      <th>geoNetwork_region</th>
      <th>geoNetwork_subContinent</th>
      <th>totals_bounces</th>
      <th>totals_hits</th>
      <th>totals_newVisits</th>
      <th>totals_pageviews</th>
      <th>totals_transactionRevenue</th>
      <th>totals_visits</th>
      <th>trafficSource_adContent</th>
      <th>trafficSource_adwordsClickInfo.adNetworkType</th>
      <th>trafficSource_adwordsClickInfo.criteriaParameters</th>
      <th>trafficSource_adwordsClickInfo.gclId</th>
      <th>trafficSource_adwordsClickInfo.isVideoAd</th>
      <th>trafficSource_adwordsClickInfo.page</th>
      <th>trafficSource_adwordsClickInfo.slot</th>
      <th>trafficSource_campaign</th>
      <th>trafficSource_campaignCode</th>
      <th>trafficSource_isTrueDirect</th>
      <th>trafficSource_keyword</th>
      <th>trafficSource_medium</th>
      <th>trafficSource_referralPath</th>
      <th>trafficSource_source</th>
      <th>dst</th>
      <th>timezoneId</th>
      <th>offset</th>
      <th>timezoneName</th>
      <th>unixtimeinzone</th>
      <th>month</th>
      <th>hour</th>
      <th>day</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Organic Search</td>
      <td>20160902</td>
      <td>377306020877927890</td>
      <td>377306020877927890_1472880147</td>
      <td>Not Socially Engaged</td>
      <td>1472880147</td>
      <td>1</td>
      <td>1472880147</td>
      <td>Firefox</td>
      <td>not available in demo dataset</td>
      <td>not available in demo dataset</td>
      <td>desktop</td>
      <td>not available in demo dataset</td>
      <td>False</td>
      <td>not available in demo dataset</td>
      <td>not available in demo dataset</td>
      <td>not available in demo dataset</td>
      <td>not available in demo dataset</td>
      <td>not available in demo dataset</td>
      <td>not available in demo dataset</td>
      <td>Macintosh</td>
      <td>not available in demo dataset</td>
      <td>not available in demo dataset</td>
      <td>not available in demo dataset</td>
      <td>not available in demo dataset</td>
      <td>not available in demo dataset</td>
      <td>Oceania</td>
      <td>Australia</td>
      <td>not available in demo dataset</td>
      <td>not available in demo dataset</td>
      <td>not available in demo dataset</td>
      <td>dodo.net.au</td>
      <td>not available in demo dataset</td>
      <td>not available in demo dataset</td>
      <td>Australasia</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>NaN</td>
      <td>1</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>not available in demo dataset</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>(not set)</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>(not provided)</td>
      <td>organic</td>
      <td>NaN</td>
      <td>google</td>
      <td>0.0</td>
      <td>Australia/Darwin</td>
      <td>34200.0</td>
      <td>Australian Central Standard Time</td>
      <td>1472914347</td>
      <td>9</td>
      <td>14</td>
      <td>3</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Organic Search</td>
      <td>20160902</td>
      <td>27294437909732085</td>
      <td>27294437909732085_1472822600</td>
      <td>Not Socially Engaged</td>
      <td>1472822600</td>
      <td>2</td>
      <td>1472822600</td>
      <td>Chrome</td>
      <td>not available in demo dataset</td>
      <td>not available in demo dataset</td>
      <td>mobile</td>
      <td>not available in demo dataset</td>
      <td>True</td>
      <td>not available in demo dataset</td>
      <td>not available in demo dataset</td>
      <td>not available in demo dataset</td>
      <td>not available in demo dataset</td>
      <td>not available in demo dataset</td>
      <td>not available in demo dataset</td>
      <td>Android</td>
      <td>not available in demo dataset</td>
      <td>not available in demo dataset</td>
      <td>not available in demo dataset</td>
      <td>not available in demo dataset</td>
      <td>not available in demo dataset</td>
      <td>Europe</td>
      <td>United Kingdom</td>
      <td>not available in demo dataset</td>
      <td>not available in demo dataset</td>
      <td>not available in demo dataset</td>
      <td>unknown.unknown</td>
      <td>not available in demo dataset</td>
      <td>not available in demo dataset</td>
      <td>Northern Europe</td>
      <td>1</td>
      <td>1</td>
      <td>NaN</td>
      <td>1</td>
      <td>NaN</td>
      <td>1</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>not available in demo dataset</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>(not set)</td>
      <td>NaN</td>
      <td>True</td>
      <td>(not provided)</td>
      <td>organic</td>
      <td>NaN</td>
      <td>google</td>
      <td>0.0</td>
      <td>Europe/London</td>
      <td>0.0</td>
      <td>Greenwich Mean Time</td>
      <td>1472822600</td>
      <td>9</td>
      <td>13</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>



### The new columns derived are below

timezoneId 	offset 	timezoneName 	unixtimeinzone 	month 	hour 	day<br>
Australian Central Standard Time 	1472914347 	9 	14 	3<br>
Europe/London 	0.0 	Greenwich Mean Time 	1472822600 	9 	13 	2<br>

These columns will be used a features .They collectiviely show the local time down to the hour


 Some data columns are cleaned and defaulted to preset values.
 
 Most the of the transactions had occured in either Google Americas or Google Rest of the world store
 Rest of world stores used GBP (Pound) sterling as currency in transation amount. hence we are bumping up transaction
 amount by approx 1.32 times for GBP transactions (We did not have this particular info , and used research and web site)
 to devise the same


```python
train_tz['visitNumber'] = train['visitNumber'].astype('int16')
train_tz['totals_hits'] = train['totals_hits'].astype('int16')
train_tz['totals_pageviews'] = train['totals_pageviews'].astype('float64')
train_tz['trafficSource_adwordsClickInfo.page'] = train['trafficSource_adwordsClickInfo.page'].astype('float64')
train_tz['totals_newVisits'] = train['totals_newVisits'].astype('float64')
train_tz['totals_bounces'] = train['totals_bounces'].astype('float64')
train_tz['totals_transactionRevenue'] = train['totals_transactionRevenue'].fillna(0).astype('int64')
train_tz['totals_transactionRevenue'] = train_tz['totals_transactionRevenue'].loc[~(train_tz.geoNetwork_country.str.upper() == 'UNITED STATES') & ~(train_tz.geoNetwork_country.str.upper() == 'CANADA')]*1.32

```


```python
train = train_tz.copy()
```


```python
def impute_source(x):
    if('googleplex' in x.lower()):
        return 'googlestore'
    elif(np.bool(re.match('^google',x.lower()))):
        return 'googlewebsite'
    elif(np.bool(re.match('.*google',x.lower()))):
        return 'googlesupport'
    elif('(direct)' in x.lower()):
        return 'direct'
    else:
        return 'nongoogle'

k=train.apply(lambda x : impute_source( (x['trafficSource_source'])), axis = 1 )
k = pd.DataFrame(k ,columns = ['sourcecat'])
train = pd.concat([train ,k] , axis =1)
```

Data observation showed traffic originated from google affiliated employee domains
Like GooglePlex ,google websites (google search engine) , google domain like gamail etc  and Direct visits to the store
These features are likely to affect the purchasing decision as well
Hence they are being imputed into categories.


The following features are finally chosen for the algorthim. These are are needed to be one hot encoded for consumption
into a lightgbm tree boost algorithm
##### channelGrouping
##### geoNetwork_continent 
#### trafficSource_medium
#### sourcecat

The function below will one hot encode the said columns



```python
def dataOneHotEncoder(df: pd.DataFrame , gbcols : list):
  for colname in gbcols:
      df = pd.concat([df,pd.get_dummies(df[colname], prefix=colname,dummy_na=True)],axis=1).drop([colname],axis=1)
      return df
```


```python
#train = dataOneHotEncoder(train , [('channelGrouping'),('geoNetwork_continent') , ('trafficSource_medium') , ('sourcecat')])
#train = dataOneHotEncoder(train , [('trafficSource_medium')])
```


```python
train.head(3)
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
      <th>date</th>
      <th>fullVisitorId</th>
      <th>sessionId</th>
      <th>socialEngagementType</th>
      <th>visitId</th>
      <th>visitNumber</th>
      <th>visitStartTime</th>
      <th>device_browser</th>
      <th>device_browserSize</th>
      <th>device_browserVersion</th>
      <th>device_deviceCategory</th>
      <th>device_flashVersion</th>
      <th>device_isMobile</th>
      <th>device_language</th>
      <th>device_mobileDeviceBranding</th>
      <th>device_mobileDeviceInfo</th>
      <th>device_mobileDeviceMarketingName</th>
      <th>device_mobileDeviceModel</th>
      <th>device_mobileInputSelector</th>
      <th>device_operatingSystem</th>
      <th>device_operatingSystemVersion</th>
      <th>device_screenColors</th>
      <th>device_screenResolution</th>
      <th>geoNetwork_city</th>
      <th>geoNetwork_cityId</th>
      <th>geoNetwork_country</th>
      <th>geoNetwork_latitude</th>
      <th>geoNetwork_longitude</th>
      <th>geoNetwork_metro</th>
      <th>geoNetwork_networkDomain</th>
      <th>geoNetwork_networkLocation</th>
      <th>geoNetwork_region</th>
      <th>geoNetwork_subContinent</th>
      <th>totals_bounces</th>
      <th>totals_hits</th>
      <th>totals_newVisits</th>
      <th>totals_pageviews</th>
      <th>totals_transactionRevenue</th>
      <th>totals_visits</th>
      <th>trafficSource_adContent</th>
      <th>trafficSource_adwordsClickInfo.adNetworkType</th>
      <th>trafficSource_adwordsClickInfo.criteriaParameters</th>
      <th>trafficSource_adwordsClickInfo.gclId</th>
      <th>trafficSource_adwordsClickInfo.isVideoAd</th>
      <th>trafficSource_adwordsClickInfo.page</th>
      <th>trafficSource_adwordsClickInfo.slot</th>
      <th>trafficSource_campaign</th>
      <th>trafficSource_campaignCode</th>
      <th>trafficSource_isTrueDirect</th>
      <th>trafficSource_keyword</th>
      <th>trafficSource_medium</th>
      <th>trafficSource_referralPath</th>
      <th>trafficSource_source</th>
      <th>dst</th>
      <th>timezoneId</th>
      <th>offset</th>
      <th>timezoneName</th>
      <th>unixtimeinzone</th>
      <th>month</th>
      <th>hour</th>
      <th>day</th>
      <th>sourcecat</th>
      <th>channelGrouping_(Other)</th>
      <th>channelGrouping_Affiliates</th>
      <th>channelGrouping_Direct</th>
      <th>channelGrouping_Display</th>
      <th>channelGrouping_Organic Search</th>
      <th>channelGrouping_Paid Search</th>
      <th>channelGrouping_Referral</th>
      <th>channelGrouping_Social</th>
      <th>channelGrouping_nan</th>
      <th>geoNetwork_continent_(not set)</th>
      <th>geoNetwork_continent_Africa</th>
      <th>geoNetwork_continent_Americas</th>
      <th>geoNetwork_continent_Asia</th>
      <th>geoNetwork_continent_Europe</th>
      <th>geoNetwork_continent_Oceania</th>
      <th>geoNetwork_continent_nan</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>20160902</td>
      <td>377306020877927890</td>
      <td>377306020877927890_1472880147</td>
      <td>Not Socially Engaged</td>
      <td>1472880147</td>
      <td>1</td>
      <td>1472880147</td>
      <td>Firefox</td>
      <td>not available in demo dataset</td>
      <td>not available in demo dataset</td>
      <td>desktop</td>
      <td>not available in demo dataset</td>
      <td>False</td>
      <td>not available in demo dataset</td>
      <td>not available in demo dataset</td>
      <td>not available in demo dataset</td>
      <td>not available in demo dataset</td>
      <td>not available in demo dataset</td>
      <td>not available in demo dataset</td>
      <td>Macintosh</td>
      <td>not available in demo dataset</td>
      <td>not available in demo dataset</td>
      <td>not available in demo dataset</td>
      <td>not available in demo dataset</td>
      <td>not available in demo dataset</td>
      <td>Australia</td>
      <td>not available in demo dataset</td>
      <td>not available in demo dataset</td>
      <td>not available in demo dataset</td>
      <td>dodo.net.au</td>
      <td>not available in demo dataset</td>
      <td>not available in demo dataset</td>
      <td>Australasia</td>
      <td>1.0</td>
      <td>1</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>not available in demo dataset</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>(not set)</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>(not provided)</td>
      <td>organic</td>
      <td>NaN</td>
      <td>google</td>
      <td>0.0</td>
      <td>Australia/Darwin</td>
      <td>34200.0</td>
      <td>Australian Central Standard Time</td>
      <td>1472914347</td>
      <td>9</td>
      <td>14</td>
      <td>3</td>
      <td>googlewebsite</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>20160902</td>
      <td>27294437909732085</td>
      <td>27294437909732085_1472822600</td>
      <td>Not Socially Engaged</td>
      <td>1472822600</td>
      <td>1</td>
      <td>1472822600</td>
      <td>Chrome</td>
      <td>not available in demo dataset</td>
      <td>not available in demo dataset</td>
      <td>mobile</td>
      <td>not available in demo dataset</td>
      <td>True</td>
      <td>not available in demo dataset</td>
      <td>not available in demo dataset</td>
      <td>not available in demo dataset</td>
      <td>not available in demo dataset</td>
      <td>not available in demo dataset</td>
      <td>not available in demo dataset</td>
      <td>Android</td>
      <td>not available in demo dataset</td>
      <td>not available in demo dataset</td>
      <td>not available in demo dataset</td>
      <td>not available in demo dataset</td>
      <td>not available in demo dataset</td>
      <td>United Kingdom</td>
      <td>not available in demo dataset</td>
      <td>not available in demo dataset</td>
      <td>not available in demo dataset</td>
      <td>unknown.unknown</td>
      <td>not available in demo dataset</td>
      <td>not available in demo dataset</td>
      <td>Northern Europe</td>
      <td>1.0</td>
      <td>1</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>not available in demo dataset</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>(not set)</td>
      <td>NaN</td>
      <td>True</td>
      <td>(not provided)</td>
      <td>organic</td>
      <td>NaN</td>
      <td>google</td>
      <td>0.0</td>
      <td>Europe/London</td>
      <td>0.0</td>
      <td>Greenwich Mean Time</td>
      <td>1472822600</td>
      <td>9</td>
      <td>13</td>
      <td>2</td>
      <td>googlewebsite</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>20160902</td>
      <td>2938943183656635653</td>
      <td>2938943183656635653_1472807194</td>
      <td>Not Socially Engaged</td>
      <td>1472807194</td>
      <td>1</td>
      <td>1472807194</td>
      <td>Chrome</td>
      <td>not available in demo dataset</td>
      <td>not available in demo dataset</td>
      <td>desktop</td>
      <td>not available in demo dataset</td>
      <td>False</td>
      <td>not available in demo dataset</td>
      <td>not available in demo dataset</td>
      <td>not available in demo dataset</td>
      <td>not available in demo dataset</td>
      <td>not available in demo dataset</td>
      <td>not available in demo dataset</td>
      <td>Windows</td>
      <td>not available in demo dataset</td>
      <td>not available in demo dataset</td>
      <td>not available in demo dataset</td>
      <td>not available in demo dataset</td>
      <td>not available in demo dataset</td>
      <td>Italy</td>
      <td>not available in demo dataset</td>
      <td>not available in demo dataset</td>
      <td>not available in demo dataset</td>
      <td>fastwebnet.it</td>
      <td>not available in demo dataset</td>
      <td>not available in demo dataset</td>
      <td>Southern Europe</td>
      <td>1.0</td>
      <td>1</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>not available in demo dataset</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>(not set)</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>(not provided)</td>
      <td>organic</td>
      <td>NaN</td>
      <td>google</td>
      <td>0.0</td>
      <td>Europe/Rome</td>
      <td>3600.0</td>
      <td>Central European Standard Time</td>
      <td>1472810794</td>
      <td>9</td>
      <td>10</td>
      <td>2</td>
      <td>googlewebsite</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



Data post encoding looks like as above . The classes below are fucntions for cross validation and data splits



```python
from sklearn.metrics import mean_squared_error
import gc
import time
from pandas.core.common import SettingWithCopyWarning
import warnings
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
warnings.simplefilter('error', SettingWithCopyWarning)
gc.enable()
def get_folds(df=None, n_splits=5):
    """Returns dataframe indices corresponding to Visitors Group KFold"""
    # Get sorted unique visitors
    unique_vis = np.array(sorted(df['fullVisitorId'].unique()))

    # Get folds
    folds = GroupKFold(n_splits=n_splits)
    fold_ids = []
    ids = np.arange(df.shape[0])
    for trn_vis, val_vis in folds.split(X=unique_vis, y=unique_vis, groups=unique_vis):
        fold_ids.append(
            [
                ids[df['fullVisitorId'].isin(unique_vis[trn_vis])],
                ids[df['fullVisitorId'].isin(unique_vis[val_vis])]
            ]
        )

    return fold_ids
```


```python
y_reg = train['totals_transactionRevenue']
folds = get_folds(df=train, n_splits=5)
importances = pd.DataFrame()
oof_reg_preds = np.zeros(train.shape[0])
```


```python
train.head(3)
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
      <th>date</th>
      <th>fullVisitorId</th>
      <th>sessionId</th>
      <th>socialEngagementType</th>
      <th>visitId</th>
      <th>visitNumber</th>
      <th>visitStartTime</th>
      <th>device_browser</th>
      <th>device_browserSize</th>
      <th>device_browserVersion</th>
      <th>device_deviceCategory</th>
      <th>device_flashVersion</th>
      <th>device_isMobile</th>
      <th>device_language</th>
      <th>device_mobileDeviceBranding</th>
      <th>device_mobileDeviceInfo</th>
      <th>device_mobileDeviceMarketingName</th>
      <th>device_mobileDeviceModel</th>
      <th>device_mobileInputSelector</th>
      <th>device_operatingSystem</th>
      <th>device_operatingSystemVersion</th>
      <th>device_screenColors</th>
      <th>device_screenResolution</th>
      <th>geoNetwork_city</th>
      <th>geoNetwork_cityId</th>
      <th>geoNetwork_country</th>
      <th>geoNetwork_latitude</th>
      <th>geoNetwork_longitude</th>
      <th>geoNetwork_metro</th>
      <th>geoNetwork_networkDomain</th>
      <th>geoNetwork_networkLocation</th>
      <th>geoNetwork_region</th>
      <th>geoNetwork_subContinent</th>
      <th>totals_bounces</th>
      <th>totals_hits</th>
      <th>totals_newVisits</th>
      <th>totals_pageviews</th>
      <th>totals_transactionRevenue</th>
      <th>totals_visits</th>
      <th>trafficSource_adContent</th>
      <th>trafficSource_adwordsClickInfo.adNetworkType</th>
      <th>trafficSource_adwordsClickInfo.criteriaParameters</th>
      <th>trafficSource_adwordsClickInfo.gclId</th>
      <th>trafficSource_adwordsClickInfo.isVideoAd</th>
      <th>trafficSource_adwordsClickInfo.page</th>
      <th>trafficSource_adwordsClickInfo.slot</th>
      <th>trafficSource_campaign</th>
      <th>trafficSource_campaignCode</th>
      <th>trafficSource_isTrueDirect</th>
      <th>trafficSource_keyword</th>
      <th>trafficSource_referralPath</th>
      <th>trafficSource_source</th>
      <th>dst</th>
      <th>timezoneId</th>
      <th>offset</th>
      <th>timezoneName</th>
      <th>unixtimeinzone</th>
      <th>month</th>
      <th>hour</th>
      <th>day</th>
      <th>channelGrouping_(Other)</th>
      <th>channelGrouping_Affiliates</th>
      <th>channelGrouping_Direct</th>
      <th>channelGrouping_Display</th>
      <th>channelGrouping_Organic Search</th>
      <th>channelGrouping_Paid Search</th>
      <th>channelGrouping_Referral</th>
      <th>channelGrouping_Social</th>
      <th>channelGrouping_nan</th>
      <th>geoNetwork_continent_(not set)</th>
      <th>geoNetwork_continent_Africa</th>
      <th>geoNetwork_continent_Americas</th>
      <th>geoNetwork_continent_Asia</th>
      <th>geoNetwork_continent_Europe</th>
      <th>geoNetwork_continent_Oceania</th>
      <th>geoNetwork_continent_nan</th>
      <th>sourcecat_direct</th>
      <th>sourcecat_googlestore</th>
      <th>sourcecat_googlesupport</th>
      <th>sourcecat_googlewebsite</th>
      <th>sourcecat_nongoogle</th>
      <th>sourcecat_nan</th>
      <th>trafficSource_medium_(none)</th>
      <th>trafficSource_medium_(not set)</th>
      <th>trafficSource_medium_affiliate</th>
      <th>trafficSource_medium_cpc</th>
      <th>trafficSource_medium_cpm</th>
      <th>trafficSource_medium_organic</th>
      <th>trafficSource_medium_referral</th>
      <th>trafficSource_medium_nan</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>20160902</td>
      <td>377306020877927890</td>
      <td>377306020877927890_1472880147</td>
      <td>Not Socially Engaged</td>
      <td>1472880147</td>
      <td>1</td>
      <td>1472880147</td>
      <td>Firefox</td>
      <td>not available in demo dataset</td>
      <td>not available in demo dataset</td>
      <td>desktop</td>
      <td>not available in demo dataset</td>
      <td>False</td>
      <td>not available in demo dataset</td>
      <td>not available in demo dataset</td>
      <td>not available in demo dataset</td>
      <td>not available in demo dataset</td>
      <td>not available in demo dataset</td>
      <td>not available in demo dataset</td>
      <td>Macintosh</td>
      <td>not available in demo dataset</td>
      <td>not available in demo dataset</td>
      <td>not available in demo dataset</td>
      <td>not available in demo dataset</td>
      <td>not available in demo dataset</td>
      <td>Australia</td>
      <td>not available in demo dataset</td>
      <td>not available in demo dataset</td>
      <td>not available in demo dataset</td>
      <td>dodo.net.au</td>
      <td>not available in demo dataset</td>
      <td>not available in demo dataset</td>
      <td>Australasia</td>
      <td>1.0</td>
      <td>1</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>not available in demo dataset</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>(not set)</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>(not provided)</td>
      <td>NaN</td>
      <td>google</td>
      <td>0.0</td>
      <td>Australia/Darwin</td>
      <td>34200.0</td>
      <td>Australian Central Standard Time</td>
      <td>1472914347</td>
      <td>9</td>
      <td>14</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>20160902</td>
      <td>27294437909732085</td>
      <td>27294437909732085_1472822600</td>
      <td>Not Socially Engaged</td>
      <td>1472822600</td>
      <td>1</td>
      <td>1472822600</td>
      <td>Chrome</td>
      <td>not available in demo dataset</td>
      <td>not available in demo dataset</td>
      <td>mobile</td>
      <td>not available in demo dataset</td>
      <td>True</td>
      <td>not available in demo dataset</td>
      <td>not available in demo dataset</td>
      <td>not available in demo dataset</td>
      <td>not available in demo dataset</td>
      <td>not available in demo dataset</td>
      <td>not available in demo dataset</td>
      <td>Android</td>
      <td>not available in demo dataset</td>
      <td>not available in demo dataset</td>
      <td>not available in demo dataset</td>
      <td>not available in demo dataset</td>
      <td>not available in demo dataset</td>
      <td>United Kingdom</td>
      <td>not available in demo dataset</td>
      <td>not available in demo dataset</td>
      <td>not available in demo dataset</td>
      <td>unknown.unknown</td>
      <td>not available in demo dataset</td>
      <td>not available in demo dataset</td>
      <td>Northern Europe</td>
      <td>1.0</td>
      <td>1</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>not available in demo dataset</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>(not set)</td>
      <td>NaN</td>
      <td>True</td>
      <td>(not provided)</td>
      <td>NaN</td>
      <td>google</td>
      <td>0.0</td>
      <td>Europe/London</td>
      <td>0.0</td>
      <td>Greenwich Mean Time</td>
      <td>1472822600</td>
      <td>9</td>
      <td>13</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>20160902</td>
      <td>2938943183656635653</td>
      <td>2938943183656635653_1472807194</td>
      <td>Not Socially Engaged</td>
      <td>1472807194</td>
      <td>1</td>
      <td>1472807194</td>
      <td>Chrome</td>
      <td>not available in demo dataset</td>
      <td>not available in demo dataset</td>
      <td>desktop</td>
      <td>not available in demo dataset</td>
      <td>False</td>
      <td>not available in demo dataset</td>
      <td>not available in demo dataset</td>
      <td>not available in demo dataset</td>
      <td>not available in demo dataset</td>
      <td>not available in demo dataset</td>
      <td>not available in demo dataset</td>
      <td>Windows</td>
      <td>not available in demo dataset</td>
      <td>not available in demo dataset</td>
      <td>not available in demo dataset</td>
      <td>not available in demo dataset</td>
      <td>not available in demo dataset</td>
      <td>Italy</td>
      <td>not available in demo dataset</td>
      <td>not available in demo dataset</td>
      <td>not available in demo dataset</td>
      <td>fastwebnet.it</td>
      <td>not available in demo dataset</td>
      <td>not available in demo dataset</td>
      <td>Southern Europe</td>
      <td>1.0</td>
      <td>1</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>not available in demo dataset</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>(not set)</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>(not provided)</td>
      <td>NaN</td>
      <td>google</td>
      <td>0.0</td>
      <td>Europe/Rome</td>
      <td>3600.0</td>
      <td>Central European Standard Time</td>
      <td>1472810794</td>
      <td>9</td>
      <td>10</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
## The features chosne for the algorithm
X= train[['totals_hits','totals_newVisits','totals_pageviews','month','hour','day','channelGrouping_(Other)','channelGrouping_Affiliates','channelGrouping_Direct','channelGrouping_Display','channelGrouping_Organic Search','channelGrouping_Paid Search','channelGrouping_Referral','channelGrouping_Social','channelGrouping_nan','geoNetwork_continent_(not set)','geoNetwork_continent_Africa','geoNetwork_continent_Americas','geoNetwork_continent_Asia','geoNetwork_continent_Europe','geoNetwork_continent_Oceania','geoNetwork_continent_nan','sourcecat_direct','sourcecat_googlestore','sourcecat_googlesupport','sourcecat_googlewebsite','sourcecat_nongoogle','sourcecat_nan','trafficSource_medium_(none)','trafficSource_medium_(not set)','trafficSource_medium_affiliate','trafficSource_medium_cpc','trafficSource_medium_cpm','trafficSource_medium_organic','trafficSource_medium_referral','trafficSource_medium_nan']]
```


```python
train_features = X.columns
#train_features
y_reg = y_reg.fillna(0)
X = X.fillna(0)
X.head(1)
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
      <th>totals_hits</th>
      <th>totals_newVisits</th>
      <th>totals_pageviews</th>
      <th>month</th>
      <th>hour</th>
      <th>day</th>
      <th>channelGrouping_(Other)</th>
      <th>channelGrouping_Affiliates</th>
      <th>channelGrouping_Direct</th>
      <th>channelGrouping_Display</th>
      <th>channelGrouping_Organic Search</th>
      <th>channelGrouping_Paid Search</th>
      <th>channelGrouping_Referral</th>
      <th>channelGrouping_Social</th>
      <th>channelGrouping_nan</th>
      <th>geoNetwork_continent_(not set)</th>
      <th>geoNetwork_continent_Africa</th>
      <th>geoNetwork_continent_Americas</th>
      <th>geoNetwork_continent_Asia</th>
      <th>geoNetwork_continent_Europe</th>
      <th>geoNetwork_continent_Oceania</th>
      <th>geoNetwork_continent_nan</th>
      <th>sourcecat_direct</th>
      <th>sourcecat_googlestore</th>
      <th>sourcecat_googlesupport</th>
      <th>sourcecat_googlewebsite</th>
      <th>sourcecat_nongoogle</th>
      <th>sourcecat_nan</th>
      <th>trafficSource_medium_(none)</th>
      <th>trafficSource_medium_(not set)</th>
      <th>trafficSource_medium_affiliate</th>
      <th>trafficSource_medium_cpc</th>
      <th>trafficSource_medium_cpm</th>
      <th>trafficSource_medium_organic</th>
      <th>trafficSource_medium_referral</th>
      <th>trafficSource_medium_nan</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>9</td>
      <td>14</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
oof_reg_preds = np.zeros(train.shape[0])
##sub_reg_preds = np.zeros(test.shape[0])
for fold_, (trn_, val_) in enumerate(folds):
    trn_x, trn_y = train[train_features].iloc[trn_], y_reg.iloc[trn_]
    val_x, val_y = train[train_features].iloc[val_], y_reg.iloc[val_]
    
    reg = lgb.LGBMRegressor(
        num_leaves=31,
        learning_rate=0.03,
        n_estimators=1000,
        subsample=.9,
        colsample_bytree=.9,
        random_state=6
    )
    reg.fit(
        trn_x, np.log1p(trn_y),
        eval_set=[(val_x, np.log1p(val_y))],
        early_stopping_rounds=50,
        verbose=100,
       categorical_feature = ['month','hour','day','channelGrouping_(Other)','channelGrouping_Affiliates','channelGrouping_Direct','channelGrouping_Display','channelGrouping_Organic Search','channelGrouping_Paid Search','channelGrouping_Referral','channelGrouping_Social','channelGrouping_nan','geoNetwork_continent_(not set)','geoNetwork_continent_Africa','geoNetwork_continent_Americas','geoNetwork_continent_Asia','geoNetwork_continent_Europe','geoNetwork_continent_Oceania','geoNetwork_continent_nan','trafficSource_medium_(none)','trafficSource_medium_(not set)','trafficSource_medium_affiliate','trafficSource_medium_cpc','trafficSource_medium_cpm','trafficSource_medium_organic','trafficSource_medium_referral','trafficSource_medium_nan','sourcecat_direct','sourcecat_googlestore','sourcecat_googlesupport','sourcecat_googlewebsite','sourcecat_nongoogle','sourcecat_nan'],
        eval_metric='rmse'
    )
    imp_df = pd.DataFrame()
    imp_df['feature'] = train_features
    imp_df['gain'] = reg.booster_.feature_importance(importance_type='gain')
    
    imp_df['fold'] = fold_ + 1
    importances = pd.concat(([importances, imp_df]), axis=0)
    
    oof_reg_preds[val_] = reg.predict(val_x, num_iteration=reg.best_iteration_)
    oof_reg_preds[oof_reg_preds < 0] = 0
   # _preds = reg.predict(test[train_features], num_iteration=reg.best_iteration_)
   # _preds[_preds < 0] = 0
    #sub_reg_preds += np.expm1(_preds) / len(folds)
    
mean_squared_error(np.log1p(y_reg), oof_reg_preds) ** .5
```

We found the least deviation observed was
Training until validation scores don't improve for 50 rounds.<br>
[100]	valid_0's rmse: 1.34795	valid_0's l2: 1.81696<br>
[200]	valid_0's rmse: 1.34569	valid_0's l2: 1.81089<br>
Early stopping, best iteration is:<br>
[229]	valid_0's rmse: 1.34509	valid_0's l2: 1.80925<br>

### 1.3479388206163994<br>

##  Result can be improved further. with additional rounds and new features. This result can be used further to  derive inferences on the transaction amounts

