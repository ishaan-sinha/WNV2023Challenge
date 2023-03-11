from selenium import webdriver
from selenium.webdriver.common.by import By

import time
import glob
import os
import pandas as pd

#token = 'yEfVECvHWIzrGqaiNzwaRURsToZbmSoX'
'''
creds = dict(token=token)
dtype = 'csv'

url = 'https://www.ncei.noaa.gov/cdo-web/api/v2/locations/FIPS:37'

r = requests.get(url, dtype, headers=creds)
print(r.json())
'''

id = 'csv-download'

driver = webdriver.Chrome("driver/chromedriver")
driver.get("https://www.ncei.noaa.gov/access/monitoring/climate-at-a-glance/statewide/time-series/1/tavg/all/12/1990-2022?base_prd=true&begbaseyear=1901&endbaseyear=2000")
time.sleep(5)

button = driver.find_element(By.ID, id)
driver.execute_script("arguments[0].click();", button)
'''
'''
list_of_files = glob.glob('/Users/ishaan/Downloads/*') # * means all if need specific format then *.csv
latest_file = max(list_of_files, key=os.path.getmtime)

os.rename(latest_file, '/Users/ishaan/PycharmProjects/WNV2023Challenge/fieuwhfwueih.csv')

with open("/Users/ishaan/PycharmProjects/WNV2023Challenge/fieuwhfwueih.csv", "r") as f:
    data = f.read().split("\n")
del data[1]
del data[1]
del data[1]
del data[1]
with open("/Users/ishaan/PycharmProjects/WNV2023Challenge/fieuwhfwueih.csv", "w") as f:
    f.write("\n".join(data))

noaa_data = pd.read_csv('fieuwhfwueih.csv', index_col=['Date'])
print(noaa_data)
