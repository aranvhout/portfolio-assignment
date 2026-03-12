#%%
#imports
import pandas as pd
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

dataset = pd.read_excel(r"C:\Users\aravhou\Downloads\dataset.xlsx") 

y = dataset ["DASS_total"].to_numpy()
x = dataset ["MALAD_total"].to_numpy()
x1 = dataset ["info_general_gender 0=male"].to_numpy()



#%%
######################### The linear regression
#define the model
def model(x, x1, b_0, b_1, b_2, b_3):
    y_predicted = b_0 + b_1 * x + b_2* x1 + b_3* (x*x1)
    return y_predicted

#squared loss function
def loss_function (beta, x, x1, y):
    b_0, b_1, b_2, b_3 = beta
    y_predicted = model(x, x1, b_0, b_1, b_2, b_3) 
    return np.sum((y-y_predicted)**2)

#fitting the model with optimise
initial_beta = [0, 0,0,0]
result = minimize(loss_function, initial_beta, args=(x,x1, y))
b_0_hat, b_1_hat,b_2_hat ,b_3_hat = result.x

print('B0 is', b_0_hat,
      '\nB1 is', b_1_hat,
      '\nB2 is', b_2_hat,
      '\nB3 is', b_3_hat)


#%%
#########################     plotting
x_values = np.linspace(np.min(x), np.max(x), 300) 

#split by gender to plot interaction effect
mask_male = x1 == 0
mask_female = x1 == 1

#model predictions
y_pred_male = model(x_values , np.zeros_like(x_values), b_0_hat, b_1_hat,b_2_hat ,b_3_hat)#male predictions
y_pred_female = model(x_values, np.ones_like(x_values ), b_0_hat, b_1_hat,b_2_hat ,b_3_hat)

#create male and female scatterplot
x_male = x[mask_male] #only take x_values that belong to men
y_male = y[mask_male] #only take y_values that belong to men
x_female = x[mask_female]
y_female = y[mask_female]

plt.scatter(x_female, y_female, color='pink', label='Female')
plt.scatter(x_male, y_male, color='lightblue', label='Male')

#plot the predictions as well
plt.plot(x_values , y_pred_female, color='red', label='Female fit')
plt.plot(x_values , y_pred_male, color='blue', label='Male fit')

plt.xlabel('Maladaptive coping score')
plt.ylabel('DASS-21 score')
plt.title('DASS-21 score as function of maladaptive coping')
plt.legend()

