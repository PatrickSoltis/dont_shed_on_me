#%%
import numpy as np
import matplotlib.pyplot as pyplot
from cvxpy import *
import pandas as pd

#%% Data Imports

#Apparent power values assume pf=0.95
office = pd.read_csv('SF_Office_import.csv') #Office building - not used in model
CARE = pd.read_csv('CARE_import.csv') #CARE/FERA residential
a10 = pd.read_csv('a10_import.csv') #Fire station
e1 = pd.read_csv('e1_import.csv') #Residential
mbl = pd.read_csv('MBL_import.csv') #Medical baseline ???
solar = pd.read_csv('solar_import.csv')

#%% Parameters A - Direct Objective Inputs (1-2)

D_df = pd.read_csv('nodes.csv')

#1 - Value ranking of different customer categories (5=highest priority, 1=lowest)
R = np.array([[0,0,0,0,5,4,3,2]])

#2 - Load total power for each node into D
D = D_df.iloc[:,[1,10,19,28,37,46,55,64]].to_numpy()
D_P = D_df.iloc[:,[4,13,22,31,40,49,58,67]].to_numpy()
D_Q = D_df.iloc[:,[7,16,25,34,43,52,61,70]].to_numpy()
#Index as: D[hour number, node number]

'''
#Future iterations may need demand split between critical and non-critical portions.
#The following loop provides a numbered index for all columns in D_df:

i = -1
for item in D_df.columns:
    i += 1
    print(i, item)
'''

# %% Parameters B (3-8)

#3 - solar PV generation
month = 'June' #select month of solar data (June/September/December)
hourly_gen = np.array(solar[month+' (kW)'].values)
zeros = np.zeros(240)
s_S = np.array([zeros, hourly_gen, hourly_gen, zeros, zeros, zeros, zeros, zeros]).T
#Index as: s_S[hour number, node number]

#4 - battery energy (batteries at nodes 1 & 2)
j_max = np.array([0, 9.5, 9.5, 0, 0, 0, 0, 95])
j_start = np.array([0, 8.0, 8.0, 0, 0, 0, 0, 40]) #arbitrarily chosen values

#5 - diesel fuel (diesel generator at node 3)
f_start = np.array([0, 0, 20, 0, 0, 0, 0, 0]) #arbitrarily chosen values

#6 - power ratings
b_rating = np.array([0, 25, 25, 0, 0, 0, 0, 40]) #battery
d_rating = np.array([0, 0, 10, 0, 0, 0, 0, 0])

#7 - power factors (currently is implicit 0.95 in D_df columns)
#pf = np.full((1,8), 0.95)

#8 - voltage limits
V_min = 0.95
V_max = 1.05

# %% Parameters C (9-11)

#9 - Resistance and reactance
r = np.array([0,0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
x = np.array([0,0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01])

#10 - Max current
I_max = np.array([0,1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2])

#11 - Adjacency matrix
A = np.array([[1, 1, 1, 1, 1, 1, 1, 1],
              [0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0]])
#A[i, j]=1 if i is the parent of j

#Rho - Defines parent nodes in for loops
rho = np.array([0, 0, 0, 0, 0, 0, 0, 0])

#%% Parameters D (12-)

#12 - Time step
dt = 1

#13 - Inverter efficiencies
nu_s = 1 #solar
nu_b = 1 #battery
#Efficiency already included in generator output. 
#Only matters for solar because solar input is total solar energy,
#and for batteries because energy put in and out of system needs to be in terms of system.
#Note values are not yet built into constraints

# %% Optimization Variables

#Define number of time steps
t = 1

#1 - Fraction of load served
F = Variable((8,t))

#2 - Load supplied (apparent, real, reactive)
l_S = Variable((8,t))
l_P = Variable((8,t))
l_Q = Variable((8,t))

#3 - Battery power dispatched (+) or stored (-)
b_S = Variable((8,t))
b_P = Variable((8,t))
b_Q = Variable((8,t))

#4 - Diesel power generated
d_S = Variable((8,t))
d_P = Variable((8,t))
d_Q = Variable((8,t))

#5 - Solar power (real and reactive)
s_P = Variable((8,t))
s_Q = Variable((8,t))
#Currently solar only provides real power: s_P = s_S (in constraints)
#So these are duplicative, but relevant if modeling inverter that can provide P and Q

#6 - Net power
s = Variable((8,t))
p = Variable((8,t))
q = Variable((8,t))

#7 - Voltage
V = Variable((8,t))

#8 - Energy stock
j = Variable((8,t)) #Battery state of charge
f = Variable((8,t)) #Diesel fuel available

#9-11 - Line variable
P = Variable((8,t)) #active power flow
Q = Variable((8,t)) #reactive power flow
L = Variable((8,t)) #squared magnitude of complex current

'''
#9-11 - Line  variables
P = Variable((8,8,t)) #active power flow
Q = Variable((8,8,t)) #reactive power flow
L = Variable((8,8,t)) #squared magnitude of complex current
# ValueError: Expressions of dimension greater than 2 are not supported.
# Alternative version (above) squeezes into 2 dimensions since all nodes 1-7 have parent 0. 
'''

# %%
