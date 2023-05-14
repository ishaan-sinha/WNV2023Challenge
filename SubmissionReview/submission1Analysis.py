import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import plotly.express as px


us_state_to_abbrev = {
    "Alabama": "AL",
    "Alaska": "AK",
    "Arizona": "AZ",
    "Arkansas": "AR",
    "California": "CA",
    "Colorado": "CO",
    "Connecticut": "CT",
    "Delaware": "DE",
    "Florida": "FL",
    "Georgia": "GA",
    "Hawaii": "HI",
    "Idaho": "ID",
    "Illinois": "IL",
    "Indiana": "IN",
    "Iowa": "IA",
    "Kansas": "KS",
    "Kentucky": "KY",
    "Louisiana": "LA",
    "Maine": "ME",
    "Maryland": "MD",
    "Massachusetts": "MA",
    "Michigan": "MI",
    "Minnesota": "MN",
    "Mississippi": "MS",
    "Missouri": "MO",
    "Montana": "MT",
    "Nebraska": "NE",
    "Nevada": "NV",
    "New Hampshire": "NH",
    "New Jersey": "NJ",
    "New Mexico": "NM",
    "New York": "NY",
    "North Carolina": "NC",
    "North Dakota": "ND",
    "Ohio": "OH",
    "Oklahoma": "OK",
    "Oregon": "OR",
    "Pennsylvania": "PA",
    "Rhode Island": "RI",
    "South Carolina": "SC",
    "South Dakota": "SD",
    "Tennessee": "TN",
    "Texas": "TX",
    "Utah": "UT",
    "Vermont": "VT",
    "Virginia": "VA",
    "Washington": "WA",
    "West Virginia": "WV",
    "Wisconsin": "WI",
    "Wyoming": "WY",
    "District of Columbia": "DC",
    "American Samoa": "AS",
    "Guam": "GU",
    "Northern Mariana Islands": "MP",
    "Puerto Rico": "PR",
    "United States Minor Outlying Islands": "UM",
    "U.S. Virgin Islands": "VI",
}

submission = pd.read_csv('../submissions/finalSubmissionCombined2.csv')
'''
for month in set(submission['target_end_date']):
    submission_month = submission[submission['target_end_date'] == month][submission['quantile'] == 0.5]
    submission_month['location'] = submission_month['location'].apply(lambda x: us_state_to_abbrev[x])
    fig = px.choropleth(submission_month, locations= submission_month['location'], locationmode="USA-states", scope="usa", color = 'value', color_continuous_scale="Viridis_r", title=month)
    fig.update_layout(
        title_text='WNV Predicted State Counts for ' + month,
        title_font_family="Times New Roman",
        title_font_size=22,
        title_font_color="black",
        title_x=0.45,
    )
    fig.write_image('submission1Graphs/' + month + '.png')
'''

for month in set(submission['target_end_date']):
    submission_month = submission[submission['target_end_date'] == month][submission['quantile'] == 0.5]
    submission_month['location'] = submission_month['location'].apply(lambda x: us_state_to_abbrev[x])
    submission_month['value'].plot.hist(bins = range(0, 80+4, 4), title = month)
    #plt.show()
    plt.savefig('submission1Graphs/' + month + 'Histogram.png')
    plt.clf()

'''
totalCounts = []
for month in sorted(list(set(submission['target_end_date']))):
    submission_month = submission[submission['target_end_date'] == month][submission['quantile'] == 0.5]
    total_count = sum(submission_month['value'])
    totalCounts.append(total_count)
plt.plot([5, 6, 7, 8, 9, 10, 11, 12], totalCounts)
plt.title('National Predicted Cases by Month')
plt.savefig('submission1Graphs/NationalPredictedCases.png')

SecondtotalCounts = []
total = 0
for month in sorted(list(set(submission['target_end_date']))):
    submission_month = submission[submission['target_end_date'] == month][submission['quantile'] == 0.5]
    monthCount = sum(submission_month['value'])
    total += monthCount
    SecondtotalCounts.append(total)
plt.plot([5, 6, 7, 8, 9, 10, 11, 12], SecondtotalCounts)
plt.title('Cumulative National Predicted Cases by Month')
plt.savefig('submission1Graphs/CumulativeNationalPredictedCases.png')
'''