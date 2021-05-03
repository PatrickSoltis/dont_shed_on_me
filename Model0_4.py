#%%
import numpy as np
import matplotlib.pyplot as pyplot
from cvxpy import *
import pandas as pd

#%% DATA IMPORTS

#Apparent power values assume pf=0.95
office = pd.read_csv('SF_Office_import.csv') #Office building - not used in model
CARE = pd.read_csv('CARE_import.csv') #CARE/FERA residential
a10 = pd.read_csv('a10_import.csv') #Fire station
e1 = pd.read_csv('e1_import.csv') #Residential
mbl = pd.read_csv('MBL_import.csv') #Medical baseline ???
solar = pd.read_csv('solar_import.csv')

#%% PARAMETERS A (0-2)

D_df = pd.read_csv('nodes.csv')

#0 - Select time range of values from load and solar data
t0 = 12 #starting hour
len_t = 3 #number of hours

#1 - Value ranking of different customer categories (5=highest priority, 1=lowest)
R = np.array([[0, 0, 0, 0, 4, 3, 2, 1]])
#Remaining length-8 arrays are only 1 dimension. 
# They will only be used in constraints at a single time.

#2 - Total power demand at each node
D = D_df.iloc[t0:t0+len_t, [1,10,19,28,37,46,55,64]].to_numpy() #Apparent
D_P = D_df.iloc[t0:t0+len_t, [4,13,22,31,40,49,58,67]].to_numpy() #Real
D_Q = D_df.iloc[t0:t0+len_t, [7,16,25,34,43,52,61,70]].to_numpy() #Reactive
#Index as: D[hour number, node number]

# %% PARAMETERS B (3-8)

#3 - solar PV generation
month = 'June' #select month of solar data (June/September/December)
hourly_gen = np.array(solar[month+' (kW)'].values[t0:t0+len_t])
zeros = np.zeros(len_t)
s_S = np.array([zeros, hourly_gen, hourly_gen, zeros, zeros, zeros, zeros, zeros]).T
#Index as: s_S[hour number, node number]

#4 - battery energy (batteries at nodes 1 & 2, EV at 7)
j_max = np.array([0, 9.5, 9.5, 0, 0, 0, 0, 95])
j_start = np.array([0, 2.0, 2.0, 0, 0, 0, 0, 0]) #arbitrarily chosen values

#5 - diesel fuel (diesel generator at node 3)
f_start = np.array([0, 0, 0, 8.0, 0, 0, 0, 0]) #arbitrarily chosen values

#6 - power ratings
b_rating = np.array([0, 8.0, 8.0, 0, 0, 0, 0, 40]) #battery
d_rating = np.array([0, 0, 0, 8.0, 0, 0, 0, 0]) #diesel

#7 - power factors (currently is an implicit 0.95 in D_df columns)
#pf = np.full((1,8), 0.95)

#8 - voltage limits
V_min = 0.95
V_max = 1.05

# %% PARAMETERS C (9-12)

#9 - Resistance and reactance
r = np.array([0,0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
x = np.array([0,0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01])

#10 - Max current
I_max = np.array([0,1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2])

#11 - Adjacency matrix
A = np.array([[0, 1, 1, 1, 1, 1, 1, 1],
              [0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0]])
#A[i, j]=1 if i is the parent of j

#12 - Node indexing
j_idx = np.arange(8) #Index including all 8 nodes
rho = np.array([0, 0, 0, 0, 0, 0, 0, 0]) #Defines parent nodes in for loops

#%% PARAMETERS D (13-14)

#13 - Time step
dt = 1

#14 - Inverter efficiencies
nu_s = 1 #solar
nu_b = 1 #battery
#These values are not yet built into constraints. 
#No efficiency value for generator because we assume generator energy already accounts for efficiency losses.
