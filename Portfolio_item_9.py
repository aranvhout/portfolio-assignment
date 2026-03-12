# -*- coding: utf-8 -*-
from scipy.stats import norm
import pandas as pd
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import statsmodels.api as sm

dataset = pd.read_excel(r"C:\Users\aravhou\Downloads\dataset.xlsx")

#convert to numpy arrays
y = dataset ["DASS_total"].to_numpy()
malad = dataset ["MALAD_total"].to_numpy()
adap= dataset ["ADAP_total"].to_numpy()
age =dataset ["info_general_age"].to_numpy()
gender = dataset ["info_general_gender 0=male"].to_numpy()
openness = dataset ["openness"].to_numpy()
conscientiousness = dataset ["conscientiousness"].to_numpy()
extraversion =dataset ["extraversion"].to_numpy()
agreeableness = dataset ["agreeableness"].to_numpy()
neuroticism =dataset["neuroticism"].to_numpy()



#%% Intiately i started off with defining my own loss functions etc. But in this case I'd also like to report
# CI and SE for completeness, so in the end I opted for a built in function


# design matrix
X = np.column_stack([malad, age, gender, openness, conscientiousness, extraversion, agreeableness, neuroticism]) #basically 
#a matrix of each partiicpants predictors values

# intercept
X = sm.add_constant(X)

# fit iwth ols
model = sm.OLS(y, X).fit()

# print summary
print(model.summary())



#%% Unadjusted

# design matrix
X = np.column_stack([malad])

# intercept
X = sm.add_constant(X)

# fit iwth ols
model = sm.OLS(y, X).fit()

# print summary
print(model.summary())




