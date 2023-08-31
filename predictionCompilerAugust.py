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


baseline_mae = pd.read_csv('modelResults/August/baseline.csv', index_col=[1])
baselineWithNational_mae = pd.read_csv('modelResults/August/baselineWithNationalTest.csv', index_col=[1])
baselineWithNationalwithWiki_mae = pd.read_csv('modelResults/August/baselineWithNationalwithWikiTest.csv', index_col=[1])
baselineWithNationalwithWikiwithLogistic_mae = pd.read_csv('modelResults/August/withArbovirusWithNationalwithWikiwithLogisticTest.csv', index_col=[1])

negative_binomial_mae = pd.read_csv('modelResults/August/statsmodels_mae.csv', index_col=[1])['negative_binomial']
poisson_mae = pd.read_csv('modelResults/August/statsmodels_mae.csv', index_col=[1])['poisson']
zero_inflated_negative_binomial_mae = pd.read_csv('modelResults/August/statsmodels_mae.csv', index_col=[1])['zero_inflated_negative_binomial']
zero_inflated_poisson_mae = pd.read_csv('modelResults/August/statsmodels_mae.csv', index_col=[1])['zero_inflated_poisson']



for state in [i for i in wnv_data['state'].unique() if i != 'DC']:
#for state in ['AL']:

    baseline = pd.read_csv('statesAugustSubmission/'+ state + '/FORECASTbaseline.csv', index_col=[0])
    baselineWithNational = pd.read_csv('statesAugustSubmission/'+ state + '/FORECASTbaselineWithNational.csv', index_col=[0])
    baselineWithNationalwithWiki = pd.read_csv('statesAugustSubmission/'+ state + '/FORECASTbaselineWithNationalwithWiki.csv', index_col=[0])
    baselineWithNationalwithWikiwithLogistic = pd.read_csv('statesAugustSubmission/'+ state + '/FORECASTbaselineWithNationalwithWikiwithLogistic.csv', index_col=[0])

    negative_binomial = pd.read_csv('statesAugustSubmission/'+ state + '/FORECASTnegative_binomial.csv', index_col=[0])
    poisson = pd.read_csv('statesAugustSubmission/'+ state + '/FORECASTpoisson.csv', index_col=[0])
    try:
        zero_inflated_negative_binomial = pd.read_csv('statesAugustSubmission/'+ state + '/FORECASTzero_inflated_negative_binomial.csv', index_col=[0])
        zero_inflated_negative_binomial.index = pd.to_datetime(zero_inflated_negative_binomial.index)
    except:
        pass
    zero_inflated_poisson = pd.read_csv('statesAugustSubmission/'+ state + '/FORECASTzero_inflated_poisson.csv', index_col=[0])

    baseline.index = pd.to_datetime(baseline.index)
    baselineWithNational.index = pd.to_datetime(baselineWithNational.index)
    baselineWithNationalwithWiki.index = pd.to_datetime(baselineWithNationalwithWiki.index)
    baselineWithNationalwithWikiwithLogistic.index = pd.to_datetime(baselineWithNationalwithWikiwithLogistic.index)

    negative_binomial.index = pd.to_datetime(negative_binomial.index)
    poisson.index = pd.to_datetime(poisson.index)
    zero_inflated_poisson.index = pd.to_datetime(zero_inflated_poisson.index)


    import calendar
    from datetime import datetime
    for ind in baseline.index:
        for col in baseline.columns:

            baselineMAE = baseline_mae.loc[state, 'baseline']
            baselineWithNationalMAE = baselineWithNational_mae.loc[state, 'withNational']
            baselineWithNationalwithWikiMAE = baselineWithNationalwithWiki_mae.loc[state, 'withNationalandWiki']
            baselineWithNationalwithWikiwithLogisticMAE = baselineWithNationalwithWikiwithLogistic_mae.loc[state, 'withNationalandWikiandLogistic']

            negative_binomialMAE = negative_binomial_mae.loc[state]
            poissonMAE = poisson_mae.loc[state]
            zero_inflated_negative_binomialMAE = zero_inflated_negative_binomial_mae.loc[state]
            zero_inflated_poissonMAE = zero_inflated_poisson_mae.loc[state]

            smallest = min(baselineMAE, baselineWithNationalMAE, baselineWithNationalwithWikiMAE, baselineWithNationalwithWikiwithLogisticMAE, negative_binomialMAE, poissonMAE, zero_inflated_negative_binomialMAE, zero_inflated_poissonMAE)

            if(smallest == baselineMAE):
                value = baseline.loc[ind, col]
            elif(smallest == baselineWithNationalMAE):
                value = baselineWithNational.loc[ind, col]
            elif(smallest == baselineWithNationalwithWikiMAE):
                value = baselineWithNationalwithWiki.loc[ind, col]
            elif(smallest == baselineWithNationalwithWikiwithLogisticMAE):
                value = baselineWithNationalwithWikiwithLogistic.loc[ind, col]
            elif(smallest == negative_binomialMAE):
                value = negative_binomial.loc[ind, col]
            elif(smallest == poissonMAE):
                value = poisson.loc[ind, col]
            elif(smallest == zero_inflated_negative_binomialMAE):
                value = zero_inflated_negative_binomial.loc[ind, col]
            elif(smallest == zero_inflated_poissonMAE):
                value = zero_inflated_poisson.loc[ind, col]

            value = max(0, value)
            toConcat = pd.DataFrame({'location': abbrev_to_us_state.get(state), 'forecast_date': '2023-07-31', 'target_end_date': str(ind.year) + '-' + f"{ind.month:02}" + '-' + str(calendar.monthrange(ind.year, ind.month)[1]), 'target': calendar.month_name[ind.month] + " WNV neuroinvasive disease cases" , 'type': 'quantile', 'quantile': int(col)/1000, 'value': value}, index=[0])
            finalSubmission = pd.concat([finalSubmission, toConcat], ignore_index=True)
for state in ['DC']:
    third_pred = pd.read_csv('statesAugustSubmission/'+ state + '/FORECASTbaseline.csv', index_col=[0])
    third_pred.index = pd.to_datetime(third_pred.index)
    for ind in third_pred.index:
        for col in third_pred.columns:
            correctDate = ind + pd.DateOffset(months=0)
            value = third_pred.loc[ind, col]
            value = max(0, value)
            toConcat = pd.DataFrame({'location': abbrev_to_us_state.get(state), 'forecast_date': '2023-07-31', 'target_end_date': str(correctDate.year) + '-' + f"{correctDate.month:02}" + '-' + str(calendar.monthrange(correctDate.year, correctDate.month)[1]), 'target': calendar.month_name[correctDate.month] + " WNV neuroinvasive disease cases" , 'type': 'quantile', 'quantile': int(col)/1000, 'value': value}, index=[0])
            finalSubmission = pd.concat([finalSubmission, toConcat], ignore_index=True)

finalSubmission.to_csv('submissions/finalSubmissionJuly.csv')
