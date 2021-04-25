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
R = np.array([[0, 0, 0, 0, 5, 4, 3, 2]])
#Remaining length-8 arrays are only 1 dimension. 
# They will only be used in constraints at a single time.

#Select hour numbers to use from load and solar data
t0 = 12
len_t = 1

#2 - Load total power for each node into D
D = D_df.iloc[t0:t0+len_t, [1,10,19,28,37,46,55,64]].to_numpy()
D_P = D_df.iloc[t0:t0+len_t, [4,13,22,31,40,49,58,67]].to_numpy()
D_Q = D_df.iloc[t0:t0+len_t, [7,16,25,34,43,52,61,70]].to_numpy()
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
zeros = np.zeros(len_t)
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

j_idx = np.arange(8)

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
#Variable shape: (rows, columns)

#1 - Fraction of load served
F = Variable((len_t,8))

#2 - Load supplied (apparent, real, reactive)
l_S = Variable((len_t,8))
l_P = Variable((len_t,8))
l_Q = Variable((len_t,8))

#3 - Battery power dispatched (+) or stored (-)
b_S = Variable((len_t,8))
b_P = Variable((len_t,8))
b_Q = Variable((len_t,8))

#4 - Diesel power generated
d_S = Variable((len_t,8))
d_P = Variable((len_t,8))
d_Q = Variable((len_t,8))

#5 - Solar power (real and reactive)
s_P = Variable((len_t,8))
s_Q = Variable((len_t,8))
#Currently solar only provides real power: s_P = s_S (in constraints)
#So these are duplicative, but relevant if modeling inverter that can provide P and Q

#6 - Net power
s = Variable((len_t,8))
p = Variable((len_t,8))
q = Variable((len_t,8))

#7 - Voltage
V = Variable((len_t,8))

#8 - Energy stock
j = Variable((len_t,8)) #Battery state of charge
f = Variable((len_t,8)) #Diesel fuel available

#9-11 - Line variable
P = Variable((len_t,8)) #active power flow
Q = Variable((len_t,8)) #reactive power flow
L = Variable((len_t,8)) #squared magnitude of complex current

'''
#9-11 - Line  variables
P = Variable((8,8,t)) #active power flow
Q = Variable((8,8,t)) #reactive power flow
L = Variable((8,8,t)) #squared magnitude of complex current
# ValueError: Expressions of dimension greater than 2 are not supported.
# Alternative version (above) squeezes into 2 dimensions since all nodes 1-7 have parent 0. 
'''

# %% Define objective function

objective = Maximize( sum(R.T@F) )

# %% Constraints 0

#Nothing between node 0 and itself
constraints = [ P[0] == 0, Q[0] == 0, L[0] == 0]
#Fix node 0 voltage to be 1 p.u.
constraints += [ V[0] == 1 ]

# %% Constraints A (1-4)

#1 - Net power consumed at each node
constraints += [ p == l_P-b_P-d_P-s_P ]
constraints += [ q == l_Q-b_Q-d_Q-s_Q ]

#2 - No phantom batteries or generators
no_batteries = [0, 3, 4, 5, 6, 7]
no_generator = [0, 1, 2, 4, 5, 6, 7]
for node in no_batteries:
    constraints += [ b_S[:, node] == 0 ]
for node in no_generator:
    constraints += [ d_S[:, node] == 0 ]

#3 - Define fraction of load served (may need to loop?)
constraints += [ F == l_S/D ]

#4 - Guarantee full load for critical nodes
critical = [4, 5]
for node in critical:
    constraints += [ F[:, node] == 1. ]

# %% Constraints B (5-6)

#5 - Battery state of charge
constraints += [ j[0] == j_start ]
if len_t > 1:
    for t in range(len_t):
        constraints += [ j[t+1] == j[t] - b_S[t]*dt]
for t in range(len_t):
    constraints += [ 0 <= j[t], j[t] <= j_max]

#6 - Fuel stock
constraints += [ f[0] == f_start ]
if len_t > 1:
    for t in range(len_t):
        constraints += [ f[t+1] == f[t] - d_S[t]*dt ]

# %% Constraints C (7-9, 14)

for t in range(len_t):
    for jj in j_idx:
        i = rho[jj]

        #7 - DistFlow equations
        constraints += [ P[t,jj] == p[t,jj] + r[jj]*L[t,jj] + A[jj]@P[t,:] ]
        constraints += [ Q[t,jj] == q[t,jj] + x[jj]*L[t,jj] + A[jj]@Q[t,:] ]

        #8 - Voltage drop
        constraints += [ V[t,jj] - V[t,i] == (r[jj]**2 + x[jj]**2)*L[t,jj] - 2*(r[jj]*P[t,jj].T + x[jj]*Q[t,jj].T) ]

        #9 - Squared current magnitude (relaxed)
        constraints += [ quad_over_lin(vstack([P[t,jj],Q[t,jj]]), V[t,jj]) <= L[t,jj] ]

        #14 - Definition of apparent power
        constraints += [ norm(vstack([p[t,jj],q[t,jj]])) <= s[t,jj] ]
        #Homework only checked this relationship generation, this is for net power

# %%

D[1]
# %% Constraints D (10)

#10 - Battery and solar only emit real power
constraints += [ s_P == s_S ]
constraints += [ b_P == b_S ]

# %% Constraints E (11-13)

for t in range(len_t):
    
    #11 - Battery (dis)charging limit
    constraints += [ -b_rating <= b_S[t], b_S[t] <= b_rating ]
    constraints += [ 0 <= b_Q[t] ]

    #12 - Generator output limit
    constraints += [ b_S[t] <= d_rating ]

#13 - Battery or generator does not discharge more than available
if len_t > 1:
    for t in range(len_t):
        constraints += [ b_S[t+1]*dt <= j[t] ]
        constraints += [ d_S[t+1]*dt <= f[t] ]

# %% Constraints F (15-16)

for t in range(len_t):

    #15 - Allowed voltages
    constraints += [ V_min**2 <= V[t], V[t] <= V_max**2 ]

#16 - Current limit
constraints += [ L[t] <= I_max ]
# %%
