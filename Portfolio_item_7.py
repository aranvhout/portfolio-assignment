#%%
#imports
import pandas as pd
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

dataset = pd.read_excel(r"C:\Users\aravhou\Downloads\dataset.xlsx")

#convert to numpy arrays
y = dataset["DASS_total"].to_numpy()
x = dataset["MALAD_total"].to_numpy()
x1 = dataset["info_general_gender 0=male"].to_numpy()

#%%
#########################  Linear regression
#define the model
def model_1(x, x1, b_0, b_1, b_2, b_3, b_4):
    y_predicted = b_0 + b_1 * x + b_2 * (x**2) + b_3 * x1 + b_4 * (x * x1)#model with quadratic term
    return y_predicted

#squared loss function
def loss_function (beta, x, x1, y):
    b_0, b_1, b_2, b_3, b_4 = beta
    y_predicted = model_1(x, x1, b_0, b_1, b_2, b_3, b_4) 
    return np.sum((y-y_predicted)**2)

#fitting the model with optimise
initial_beta = [0, 0,0,0,0]
result_1 = minimize(loss_function, initial_beta, args=(x,x1, y))
b_0_hat, b_1_hat,b_2_hat ,b_3_hat, b_4_hat = result_1.x

print( 'Model 1'
      '\nB0 is', b_0_hat,
      '\nB1 is', b_1_hat,
      '\nB2 is', b_2_hat,
      '\nB3 is', b_3_hat,
      '\nB4 is', b_4_hat)


#%% plotting the linear regression with the quadratic term
x_values = np.linspace(np.min(x), np.max(x), 300) 

#split by gender to plot interaction effect
mask_male = x1 == 0
mask_female = x1 == 1

#model predictions
y_pred_male = model_1(x_values , np.zeros_like(x_values ), b_0_hat, b_1_hat,b_2_hat ,b_3_hat, b_4_hat )
y_pred_female = model_1(x_values, np.ones_like(x_values ), b_0_hat, b_1_hat,b_2_hat ,b_3_hat, b_4_hat )

#create male and female scatterplot
x_male = x[mask_male] #only take x_values that belong to men
y_male = y[mask_male]#only take y_values that belong to men
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


#%%
##### Probalistic model
def model_2(x, x1,b_0, b_1, b_2, b_3):
    y_predicted = b_0 + b_1 * x  + b_2* x1 + b_3 * (x*x1)
    return y_predicted

def NLL (beta, x, x1, y):
    b_0, b_1, b_2, b_3, sigma = beta 
    mu = model_2(x, x1,b_0, b_1, b_2, b_3)    
    n = len(y)     
  
    nll = 0.5 * np.sum(((y - mu)/sigma)**2) + n * np.log(sigma)#formule van college
    
    return nll

#fit the model
initial_beta= np.zeros(5)

bounds = [
    (None, None),  
    (None, None),  
    (None, None),  
    (None, None),        
    (1e-6, None)   # sigma should be larger than 0 otherwise the fitting can break down
]

result_2 = minimize(
    NLL,
    initial_beta,
    args=(x, x1, y),
    method='L-BFGS-B',
    bounds=bounds,
    options={'maxiter': 1000}
)

 
b_0_hat, b_1_hat,b_2_hat ,b_3_hat, sigma_hat = result_2.x

print('Model 2'
      '\nB0 is', b_0_hat,
      '\nB1 is', b_1_hat,
      '\nB2 is', b_2_hat,
      '\nB3 is', b_3_hat,
      '\n Sigma is', sigma_hat)

#%%
##### Bootstrapping
bootstrap_model_1 = []
bootstrap_model_2 = []
n_bootstrap_samples = 1000

for i in range(n_bootstrap_samples):
    indices = np.random.choice(len(x), size=len(x), replace=True)#draw len(x) indices in range len(x) with replacement
    
    x_boot_data = x[indices]
    x1_boot_data = x1[indices]
    y_boot_data = y[indices]
    
    #fit the linear regression model with the quadratic term on the bootstrapped data
    initial_beta = [0, 0,0,0,0]
    result_1 = minimize(loss_function, initial_beta, args=(x_boot_data,x1_boot_data, y_boot_data))    
    bootstrap_model_1 .append(result_1.x)
    
    #fit the prob model on the bootstrapped data
    initial_beta= np.zeros(5)
    bounds = [
        (None, None),  
        (None, None),  
        (None, None),  
        (None, None),        
        (1e-6, None)   # sigma should be larger than 0 otherwise it breaks down
    ]
    result_2 = minimize(
        NLL,
        initial_beta,
        args=(x_boot_data,x1_boot_data, y_boot_data),
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': 1000}
    )

    bootstrap_model_2 .append(result_2.x)
    
#calculate the confidence intervalse
bootstrap_model_1 = np.array(bootstrap_model_1)
bootstrap_model_2 = np.array(bootstrap_model_2)

#model 1
param_names_model_1 = ['B0', 'B1', 'B2', 'B3', 'B4']
print("Model 1:")
for i, name in enumerate(param_names_model_1): #start for loop
    mean = np.mean(bootstrap_model_1[:, i]) #mean coefficent
    se = np.std(bootstrap_model_1[:, i])  
    ci_lower = np.percentile(bootstrap_model_1[:, i], 2.5)
    ci_upper = np.percentile(bootstrap_model_1[:, i], 97.5)
    
    print("{}: mean={:.3f}, SE={:.3f}, 95% CI=({:.3f}, {:.3f})".format(
    name, mean, se, ci_lower, ci_upper))

# model 2
param_names_model_2 = ['B0', 'B1', 'B2', 'B3', 'Sigma']
print("\nModel 2:")
for i, name in enumerate(param_names_model_2):
    mean = np.mean(bootstrap_model_2[:, i])
    se = np.std(bootstrap_model_2[:, i])  
    ci_lower = np.percentile(bootstrap_model_2[:, i], 2.5)
    ci_upper = np.percentile(bootstrap_model_2[:, i], 97.5)
    print("{}: mean={:.3f}, SE={:.3f}, 95% CI=({:.3f}, {:.3f})".format(
    name, mean, se, ci_lower, ci_upper))




#%%
#plotting quadratic term distribution vs other coefficents distribution
b0_dis = bootstrap_model_1[:, 0]  # intercept
b1_dis = bootstrap_model_1[:, 1]  # lin maladaptive coping
b2_dis = bootstrap_model_1[:, 2]  # quadratic term


fig, axes = plt.subplots(1, 2, figsize=(15, 4))

# quadriatic vs intercept
axes[0].scatter(b2_dis, b0_dis, alpha=0.6)
axes[0].set_xlabel("Quadratic Term")
axes[0].set_ylabel("Intercept")
axes[0].set_title("Quadratic Coping vs Intercept")

# quadratic vs linear
axes[1].scatter(b2_dis, b1_dis, alpha=0.6)
axes[1].set_xlabel("Quadratic Term ")
axes[1].set_ylabel("Linear Term")
axes[1].set_title("Quadratic Coping vs Linear Coping")



plt.tight_layout()
plt.show()


