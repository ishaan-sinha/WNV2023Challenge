from selenium import webdriver
from selenium.webdriver.common.by import By

import time
import glob
import os
import pandas as pd

#token = 'yEfVECvHWIzrGqaiNzwaRURsToZbmSoX'

id = 'csv-download'



driver = webdriver.Chrome("driver/chromedriver")
stateDict = {1:'AL', 2:'AZ', 3:'AR', 4:'CA', 5:'CO',6:'CT',7:'DE',8:'FL',9:'GA',10:'ID',11:'IL',12:'IN',13:'IA',14:'KS',15:'KY',16:'LA',17:'ME',18:'MD',19:'MA',20:'MI',21:'MN',22:'MS',23:'MO',24:'MT',25:'NE',26:'NV',27:'NH',28:'NJ',29:'NM',30:'NY',31:'NC',32:'ND',33:'OH',34:'OK',35:'OR',36:'PA',37:'RI',38:'SC',39:'SD',40:'TN',41:'TX',42:'UT',43:'VT',44:'VA',45:'WA',46:'WV',47:'WI',48:'WY'}
#only missing DC
for i in range(3, 4):
    driver.get("https://www.ncei.noaa.gov/access/monitoring/climate-at-a-glance/statewide/time-series/"+ str(i) +"/tmax/all/12/1990-2022?base_prd=true&begbaseyear=1901&endbaseyear=2000")
    state = stateDict[i]
    time.sleep(5)
    button = driver.find_element(By.ID, id)
    driver.execute_script("arguments[0].click();", button)
    time.sleep(5)

    list_of_files = glob.glob('/Users/ishaan/Downloads/*') # * means all if need specific format then *.csv
    latest_file = max(list_of_files, key=os.path.getmtime)
    os.rename(latest_file, '/Users/ishaan/PycharmProjects/WNV2023Challenge/fieuwhfwueih.csv')
    with open("/Users/ishaan/PycharmProjects/WNV2023Challenge/fieuwhfwueih.csv", "r") as f:
        data = f.read().split("\n")
    del data[0]
    del data[0]
    del data[0]
    del data[0]
    with open("/Users/ishaan/PycharmProjects/WNV2023Challenge/fieuwhfwueih.csv", "w") as f:
        f.write("\n".join(data))

    noaa_data = pd.read_csv('fieuwhfwueih.csv')
    noaa_data.drop(noaa_data.columns[2], axis=1, inplace=True)
    noaa_data['year'] = ((noaa_data['Date'] - noaa_data['Date'] % 100)/100).astype(int)
    noaa_data['month'] = (noaa_data['Date'] % 100).astype(int)
    noaa_data.drop(noaa_data.columns[0], axis=1, inplace=True)
    noaa_data.set_index(['year', 'month'], inplace=True)

    noaa_data.to_csv('states/' + state+'/' + state + '_maxTemp_data.csv')
    os.remove('/Users/ishaan/PycharmProjects/WNV2023Challenge/fieuwhfwueih.csv')
