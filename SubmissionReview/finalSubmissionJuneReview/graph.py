
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt


submission = pd.read_csv('../../submissions/finalSubmissionMayCombinedNEW.csv')
submission = submission[submission['quantile'] == 0.5]

total = submission.groupby(['target_end_date']).sum()

total['value'].plot()
plt.ylim(0, 700)
plt.title('June Submission: US Median Monthly Forecasts of WNV Cases')
plt.savefig('june_submission.png')