from selenium import webdriver
from selenium.webdriver.common.by import By

import time
import glob
import os
import pandas as pd
import requests
from io import StringIO


#token = 'yEfVECvHWIzrGqaiNzwaRURsToZbmSoX'

id = 'csv-download'



#driver = webdriver.Chrome("driver/chromedriver")
stateDict = {1:'AL', 2:'AZ', 3:'AR', 4:'CA', 5:'CO',6:'CT',7:'DE',8:'FL',9:'GA',10:'ID',11:'IL',12:'IN',13:'IA',14:'KS',15:'KY',16:'LA',17:'ME',18:'MD',19:'MA',20:'MI',21:'MN',22:'MS',23:'MO',24:'MT',25:'NE',26:'NV',27:'NH',28:'NJ',29:'NM',30:'NY',31:'NC',32:'ND',33:'OH',34:'OK',35:'OR',36:'PA',37:'RI',38:'SC',39:'SD',40:'TN',41:'TX',42:'UT',43:'VT',44:'VA',45:'WA',46:'WV',47:'WI',48:'WY'}
#only missing DC
for i in range(1, 49):
    state = stateDict[i]
    #driver.get("https://www.ncei.noaa.gov/access/monitoring/climate-at-a-glance/statewide/time-series/" + str(i) + "/tavg/all/1/2000-2023.csv?base_prd=true&begbaseyear=1901&endbaseyear=2000")
    api_url = "https://www.ncei.noaa.gov/access/monitoring/climate-at-a-glance/statewide/time-series/" + str(i) + "/tavg/all/1/2000-2023.csv?base_prd=true&begbaseyear=1901&endbaseyear=2000"
    response = requests.get(api_url)
    text = StringIO(response.content.decode('utf-8'))
    text = text.read()
    text= ''.join(text.splitlines(keepends=True)[4:])
    text = StringIO(text)
    tavg = pd.read_csv(text)

    api_url = "https://www.ncei.noaa.gov/access/monitoring/climate-at-a-glance/statewide/time-series/" + str(i) + "/tmax/all/1/2000-2023.csv?base_prd=true&begbaseyear=1901&endbaseyear=2000"
    response = requests.get(api_url)
    text = StringIO(response.content.decode('utf-8'))
    text = text.read()
    text= ''.join(text.splitlines(keepends=True)[4:])
    text = StringIO(text)
    tmax = pd.read_csv(text)

    api_url = "https://www.ncei.noaa.gov/access/monitoring/climate-at-a-glance/statewide/time-series/" + str(i) + "/tmin/all/1/2000-2023.csv?base_prd=true&begbaseyear=1901&endbaseyear=2000"
    response = requests.get(api_url)
    text = StringIO(response.content.decode('utf-8'))
    text = text.read()
    text= ''.join(text.splitlines(keepends=True)[4:])
    text = StringIO(text)
    tmin = pd.read_csv(text)

    api_url = "https://www.ncei.noaa.gov/access/monitoring/climate-at-a-glance/statewide/time-series/" + str(i) + "/pcp/all/1/2000-2023.csv?base_prd=true&begbaseyear=1901&endbaseyear=2000"
    response = requests.get(api_url)
    text = StringIO(response.content.decode('utf-8'))
    text = text.read()
    text= ''.join(text.splitlines(keepends=True)[4:])
    text = StringIO(text)
    pcp = pd.read_csv(text)

    api_url = "https://www.ncei.noaa.gov/access/monitoring/climate-at-a-glance/statewide/time-series/" + str(i) + "/zndx/all/1/2000-2023.csv?base_prd=true&begbaseyear=1901&endbaseyear=2000"
    response = requests.get(api_url)
    text = StringIO(response.content.decode('utf-8'))
    text = text.read()
    text= ''.join(text.splitlines(keepends=True)[3:])
    text = StringIO(text)
    zndx = pd.read_csv(text)

    tavg.drop(['Anomaly'], axis=1, inplace=True)
    tmax.drop(['Anomaly'], axis=1, inplace=True)
    tmin.drop(['Anomaly'], axis=1, inplace=True)
    pcp.drop(['Anomaly'], axis=1, inplace=True)
    zndx.drop(['Anomaly'], axis=1, inplace=True)
    tavg.set_index('Date', inplace=True)
    tmax.set_index('Date', inplace=True)
    tmin.set_index('Date', inplace=True)
    pcp.set_index('Date', inplace=True)
    zndx.set_index('Date', inplace=True)
    tavg.rename(columns={'Value': 'tavg'}, inplace=True)
    tmax.rename(columns={'Value': 'tmax'}, inplace=True)
    tmin.rename(columns={'Value': 'tmin'}, inplace=True)
    pcp.rename(columns={'Value': 'pcp'}, inplace=True)
    zndx.rename(columns={'Value': 'zndx'}, inplace=True)


    noaa_data = pd.concat([tavg, tmax, tmin, pcp, zndx], axis=1)

    noaa_data['year'] = ((noaa_data.index - noaa_data.index % 100)/100).astype(int)
    noaa_data['month'] = (noaa_data.index % 100).astype(int)
    noaa_data.set_index(['year', 'month'], inplace=True)

    noaa_data.to_csv('statesJulySubmission/' + state + '/NOAA_data.csv')
    print(state)
