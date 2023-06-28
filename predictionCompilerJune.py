import pandas as pd
import numpy as np

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

# invert the dictionary
abbrev_to_us_state = dict(map(reversed, us_state_to_abbrev.items()))

wnv_data = pd.read_csv('WNVData/WNV_forecasting_challenge_state-month_cases.csv', index_col=['year', 'month'])

finalSubmission = pd.DataFrame(columns=['location', 'forecast_date', 'target_end_date', 'target', 'type', 'quantile', 'value'])
third_mae = pd.read_csv('modelResults/June/baselineThirdPredTest.csv', index_col=[1])
withArbovirus_mae = pd.read_csv('modelResults/June/withArbovirusTest.csv', index_col=[0])
withArbovirusLocalized_mae = pd.read_csv('modelResults/June/withArbovirusLocalizedTest.csv', index_col=[1])
withArbovirusLocalizedMinMax_mae = pd.read_csv('modelResults/June/withArbovirusLocalizedTestMinMax.csv', index_col=[1])

for state in [i for i in wnv_data['state'].unique() if i != 'DC']:
#for state in ['AL']:
    third_pred = pd.read_csv('statesJuneSubmission/'+ state + '/FORECASTthirdPred.csv', index_col=[0])[1:]
    withArbovirus = pd.read_csv('statesJuneSubmission/'+ state + '/FORECASTwithArbovirus.csv', index_col=[0])[1:]
    withArbovirusLocalized = pd.read_csv('statesJuneSubmission/'+ state + '/FORECASTwithArbovirusLocalized.csv', index_col=[0])[1:]
    withArbovirusLocalizedMinMax = pd.read_csv('statesJuneSubmission/'+ state + '/FORECASTwithArbovirusLocalizedMinMax.csv', index_col=[0])[1:]

    third_pred.index = pd.to_datetime(third_pred.index)
    withArbovirus.index = pd.to_datetime(withArbovirus.index)
    withArbovirusLocalized.index = pd.to_datetime(withArbovirusLocalized.index)
    withArbovirusLocalizedMinMax.index = pd.to_datetime(withArbovirusLocalizedMinMax.index)


    import calendar
    from datetime import datetime
    for ind in third_pred.index:
        for col in third_pred.columns:
            firstMAE = third_mae.loc[state, 'baselineThirdPredTest']
            secondMAE = withArbovirus_mae.loc[state, 'withArbovirus']
            thirdMAE = withArbovirusLocalized_mae.loc[state, 'withArbovirusLocalized']
            fourthMAE = withArbovirusLocalizedMinMax_mae.loc[state, 'withArbovirusLocalizedMinMax']


            if(firstMAE < secondMAE and firstMAE < thirdMAE and firstMAE < fourthMAE):
                value = third_pred.loc[ind, col]
            elif(secondMAE < firstMAE and secondMAE < thirdMAE and secondMAE < fourthMAE):
                value = withArbovirus.loc[ind, col]
            elif(thirdMAE < firstMAE and thirdMAE < secondMAE and thirdMAE < fourthMAE):
                value = withArbovirusLocalized.loc[ind, col]
            else:
                value = withArbovirusLocalizedMinMax.loc[ind, col]

            value = max(0, value)
            toConcat = pd.DataFrame({'location': abbrev_to_us_state.get(state), 'forecast_date': '2023-05-31', 'target_end_date': str(ind.year) + '-' + f"{ind.month:02}" + '-' + str(calendar.monthrange(ind.year, ind.month)[1]), 'target': calendar.month_name[ind.month] + " WNV neuroinvasive disease cases" , 'type': 'quantile', 'quantile': int(col)/1000, 'value': value}, index=[0])
            finalSubmission = pd.concat([finalSubmission, toConcat], ignore_index=True)
for state in ['DC']:
    third_pred = pd.read_csv('statesJuneSubmission/'+ state + '/FORECASTthirdPred.csv', index_col=[0])[1:]
    third_pred.index = pd.to_datetime(third_pred.index)
    for ind in third_pred.index:
        for col in third_pred.columns:
            correctDate = ind + pd.DateOffset(months=0)
            value = third_pred.loc[ind, col]
            value = max(0, value)
            toConcat = pd.DataFrame({'location': abbrev_to_us_state.get(state), 'forecast_date': '2023-05-31', 'target_end_date': str(correctDate.year) + '-' + f"{correctDate.month:02}" + '-' + str(calendar.monthrange(correctDate.year, correctDate.month)[1]), 'target': calendar.month_name[correctDate.month] + " WNV neuroinvasive disease cases" , 'type': 'quantile', 'quantile': int(col)/1000, 'value': value}, index=[0])
            finalSubmission = pd.concat([finalSubmission, toConcat], ignore_index=True)

finalSubmission.to_csv('submissions/finalSubmissionMayCombinedNEW.csv')
