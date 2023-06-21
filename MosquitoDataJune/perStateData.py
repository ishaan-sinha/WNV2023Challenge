import pandas as pd
import numpy as np
import geopy.distance

state_codes = {
    'WA': '53', 'DE': '10', 'DC': '11', 'WI': '55', 'WV': '54', 'HI': '15',
    'FL': '12', 'WY': '56', 'PR': '72', 'NJ': '34', 'NM': '35', 'TX': '48',
    'LA': '22', 'NC': '37', 'ND': '38', 'NE': '31', 'TN': '47', 'NY': '36',
    'PA': '42', 'AK': '2', 'NV': '32', 'NH': '33', 'VA': '51', 'CO': '8',
    'CA': '6', 'AL': '1', 'AR': '5', 'VT': '50', 'IL': '17', 'GA': '13',
    'IN': '18', 'IA': '19', 'MA': '25', 'AZ': '4', 'ID': '16', 'CT': '09',
    'ME': '23', 'MD': '24', 'OK': '40', 'OH': '39', 'UT': '49', 'MO': '29',
    'MN': '27', 'MI': '26', 'RI': '44', 'KS': '20', 'MT': '30', 'MS': '28',
    'SC': '45', 'KY': '21', 'OR': '41', 'SD': '46'
}
inv_map = {v: k for k, v in state_codes.items()}

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

wnv_data = pd.read_csv('../WNVData/WNV_forecasting_challenge_state-month_cases.csv', index_col=['year', 'month'])

previous_full_data = pd.read_csv('MonthlyMosquitoData.csv', index_col=['year', 'month'])
# Read in the data
abundanceData = pd.read_csv('combinedAbundanceData.csv')
infectionData = pd.read_csv('combinedInfectionData.csv')
abundanceData.set_index(['year','month', 'statefp', 'countyfp'], inplace=True)
infectionData.set_index(['year','month', 'statefp', 'countyfp'], inplace=True)

state_coordinates = pd.read_csv('stateCoordinates.csv')
state_coordinates['state'] = state_coordinates['state'].apply(lambda x: us_state_to_abbrev[x])
state_coordinates.set_index('state', inplace=True)


for state in [i for i in wnv_data['state'].unique() if i not in ['DC']]:
#for state in ['ND']:
    combinedData = pd.concat([abundanceData, infectionData], axis=1)
    combinedData.reset_index(inplace=True)
    combinedData['state_names'] = combinedData['statefp'].apply(lambda x: inv_map[str(x)])
    print(state)
    acceptable_states = []

    state_coords = (state_coordinates.loc[state]['latitude'], state_coordinates.loc[state]['longitude'])
    for check in state_coordinates.index:
        check_coords = (state_coordinates.loc[check]['latitude'], state_coordinates.loc[check]['longitude'])
        if geopy.distance.distance(state_coords, check_coords).miles < 500:
            acceptable_states.append(check)

    state_data = combinedData[combinedData['state_names'].isin(acceptable_states)]
    #print(state_data.head(100))
    #print(state_data['state_names'].value_counts())

    state_data.drop(['statefp', 'countyfp', 'state_names'], axis=1, inplace=True)

    mean = state_data.groupby(['year','month']).mean()
    max = state_data.groupby(['year','month']).max()
    min = state_data.groupby(['year','month']).min()
    state_data = pd.concat([mean, max, min], axis=1)

    state_data.interpolate(method='linear', inplace=True)
    state_data.fillna(0, inplace=True)

    state_data.to_csv('../statesJulySubmission/' + state + '/MonthlyMosquitoData500miles.csv')

