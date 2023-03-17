import requests
import json

for year in range(1994, 2023, 10):
    headers = {'Content-type': 'application/json'}
    data = json.dumps({"seriesid": ['EIUIR','EIUIQ'],"startyear":str(year), "endyear":str(year+9)})
    #EIUIR = Monthly import price index for BEA End Use, All commodities, not seasonally adjusted
    #EIUIQ = Monthly export price index for BEA End Use, All commodities, not seasonally adjusted
    p = requests.post('https://api.bls.gov/publicAPI/v2/timeseries/data/', data=data, headers=headers)
    json_data = json.loads(p.text)
    for series in json_data['Results']['series']:
        for item in series['data']:
            print(item['year'], item['period'], item['value'])

