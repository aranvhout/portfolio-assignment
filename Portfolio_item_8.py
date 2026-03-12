#imports
import pandas as pd
import numpy as np
from scipy.optimize import minimize

dataset = pd.read_excel(r"C:\Users\aravhou\Downloads\dataset.xlsx")


#convert to numpy arrays
y = dataset["DASS_total"].to_numpy()
x = dataset["MALAD_total"].to_numpy()
x1 = dataset["neuroticism"].to_numpy()



#%% 
#########################  Model without neurostiscm
#define the model
def model_1 (x,  b_0, b_1):
    y_predicted = b_0 + b_1 * x 
    return y_predicted

#squared loss function
def NLL_1 (beta, x,  y):
    b_0, b_1, sigma = beta 
    mu = model_1(x ,b_0, b_1,)
    
    n = len(y)     
  
    nll = 0.5 * np.sum(((y - mu)/sigma)**2) + n * np.log(sigma)#formule van college
    
    return nll

#fitting the model with optimise
initial_beta= np.zeros(3)

bounds = [       
    (None, None),  
    (None, None),        
    (1e-6, None)   # sigma should be larger than 0 otherwise it breaks down
]

result_1 = minimize(
    NLL_1,
    initial_beta,
    args=(x, y),
    method='L-BFGS-B',
    bounds=bounds,
    options={'maxiter': 1000}
)

 
b_0_hat_m1, b_1_hat_m1, sigma_hat_m1 = result_1.x

#calculate r squared
y_predicted_m1 = model_1(x, b_0_hat_m1, b_1_hat_m1)

SSR = np.sum((y_predicted_m1 - y)**2)
SST = np.sum((np.mean(y) - y)**2)


r_squared_m1 = 1 - (SSR/SST)
NLL1 = result_1.fun

print('\nB0 is', b_0_hat_m1,
      '\nB1 is', b_1_hat_m1,          
      '\n Sigma is', sigma_hat_m1,
      '\n R2 is', r_squared_m1,
      '\n NLL is', NLL1)



#%% 

#########################  Model with neurotoscism included
#define the model
def model_2 (x, x1, b_0, b_1, b_2):
    y_predicted = b_0 + b_1 * x + b_2 * x1
    return y_predicted

#squared loss function
def NLL_2 (beta, x, x1, y):
    b_0, b_1, b_2, sigma = beta 
    mu = model_2(x, x1,b_0, b_1, b_2)
    
    n = len(y)     
  
    nll = 0.5 * np.sum(((y - mu)/sigma)**2) + n * np.log(sigma)#formule van college
    
    return nll

#fitting the model with optimise
initial_beta= np.zeros(4)

bounds = [    
   (None, None),
    (None, None),  
    (None, None),        
    (1e-6, None)   # sigma should be larger than 0 otherwise it breaks down
]

result_2 = minimize(
    NLL_2,
    initial_beta,
    args=(x, x1, y),
    method='L-BFGS-B',
    bounds=bounds,
    options={'maxiter': 1000}
)

 
b_0_hat_m2, b_1_hat_m2,b_2_hat_m2 , sigma_hat_m2 = result_2.x

#calculate r squared
y_predicted_m2 = model_2(x, x1,b_0_hat_m2, b_1_hat_m2,b_2_hat_m2 )

SSR = np.sum((y_predicted_m2 - y)**2)
SST = np.sum((np.mean(y) - y)**2)


r_squared_m2 = 1 - (SSR/SST)

#NLL
NLL2 = result_2.fun
print('\nB0 is', b_0_hat_m2,
      '\nB1 is', b_1_hat_m2,
      '\nB2 is', b_2_hat_m2,      
      '\n Sigma is', sigma_hat_m2,
      '\n R2 is', r_squared_m2,
      '\n NLL is', NLL2)

