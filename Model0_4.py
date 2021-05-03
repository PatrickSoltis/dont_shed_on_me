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

# %% Optimization Variables
#Variable shape: (rows, columns)

#1 - Fraction of real load served
F_P = Variable((len_t,8))

#2 - Load supplied (apparent, real, reactive)
l_P = Variable((len_t,8))
l_Q = Variable((len_t,8))

#3 - Battery power dispatched (+) or stored (-)
b_gen = Variable((len_t,8)) #apparent power term
b_eat = Variable((len_t,8)) #real power term

#4 - Diesel power generated
d_S = Variable((len_t,8))

#5 - Solar power (real and reactive)
#s_P = Variable((len_t,8))
#s_Q = Variable((len_t,8))
#b_P = Variable((len_t,8))
#b_Q = Variable((len_t,8))
#Solar and battery terms input currently in terms of reactive power (S),
#but can create real and reactive variables to track split or constrain to full real power output.

#6 - Net power
s = Variable((len_t,8))
p = Variable((len_t,8))
q = Variable((len_t,8))
s_max = Variable((len_t,8))

#7 - Voltage
V = Variable((len_t,8))

#8 - Energy stock
j = Variable((len_t,8)) #Battery state of charge
f = Variable((len_t,8)) #Diesel fuel available

#9-11 - Line variable
P = Variable((len_t,8)) #active power flow
Q = Variable((len_t,8)) #reactive power flow
L = Variable((len_t,8)) #squared magnitude of complex current

# %% Define objective function

objective = Maximize( sum(F_P@R.T) )

# %% CONSTRAINTS 0

#0.1 - Nothing between node 0 and itself
constraints = [ P[:,0] == 0, Q[:,0] == 0, L[:,0] == 0]
#0.2 - Fix node 0 voltage to be 1 p.u.
constraints += [ V[:,0] == 1 ]

# %% CONSTRAINTS A (1-5)

#1 - Net power generated at each node
constraints += [ s_max == b_gen+d_S+s_S ]

#2 - No phantom batteries, generators, or loads
no_batteries = [0, 3, 4, 5, 6]
no_generator = [0, 1, 2, 4, 5, 6, 7]
no_load = [0, 1, 2, 3]
for node in no_batteries:
    constraints += [ b_gen[:, node] == 0 ]
    constraints += [ b_eat[:, node] == 0 ]
for node in no_generator:
    constraints += [ d_S[:, node] == 0 ]
for node in no_load:
    constraints += [ l_P[:, node] == 0 ]
    constraints += [ l_Q[:, node] == 0]

#3 - Define fraction of load served
constraints += [ F_P == l_P/D_P ]

#4 - Guarantee full load for critical nodes (removed for initial run)
#critical = [4, 5]
#for node in critical:
#    constraints += [ F_P[:, node] == 1. ]

# %% CONSTRAINTS B (5-6)

#5 - Battery state of charge
constraints += [ j[0] == j_start ]
if len_t > 1:
    for t in range(1,len_t):
        constraints += [ j[t] == j[t-1] - b_gen[t-1]*dt + b_eat[t-1] ]
for t in range(len_t):
    constraints += [ 0 <= j[t], j[t] <= j_max]

#6 - Fuel stock
constraints += [ f[0] == f_start ]
if len_t > 1:
    for t in range(1, len_t):
        constraints += [ f[t] == f[t-1] - d_S[t-1]*dt ]

# %% CONSTRAINTS C (7-9, 14)

for t in range(len_t):
    for jj in j_idx:
        i = rho[jj]

        #7 - DistFlow equations
        constraints += [ P[t,jj] == l_P[t,jj] + b_eat[t,jj] - p[t,jj] + r[jj]*L[t,jj] + A[jj]@P[t,:] ]
        constraints += [ Q[t,jj] == l_Q[t,jj] - q[t,jj] + x[jj]*L[t,jj] + A[jj]@Q[t,:] ]

        #8 - Voltage drop
        constraints += [ V[t,jj] - V[t,i] == (r[jj]**2 + x[jj]**2)*L[t,jj] - 2*(r[jj]*P[t,jj].T + x[jj]*Q[t,jj].T) ]

        #9 - Squared current magnitude (relaxed)
        constraints += [ quad_over_lin(vstack([P[t,jj],Q[t,jj]]), V[t,jj]) <= L[t,jj] ]

        #14 - Definition of apparent power
        constraints += [ norm(vstack([p[t,jj],q[t,jj]])) <= s[t,jj] ]
        #Homework only checked this relationship generation, this is for net power

# %% Constraints D (10-12)

#10 - Battery and solar only emit real power
#constraints += [ s_P == s_S ]
#constraints += [ b_P == b_gen ]
#s_P and b_P terms currently commented out

#11 - Power delivered cannot exceed demand
constraints += [ l_P <= D_P, 0 <= l_P ] #try without this first

#12 - Need to limit apparent power
constraints += [ s <= s_max ]
#where s is apparent power generated at each node, and s is theoretical capacity

# %% Constraints E (13-15)

for t in range(len_t):
    
    #13 - Battery (dis)charging limit
    constraints += [ 0 <= b_gen[t], b_gen[t] <= b_rating ] #discharging
    constraints += [ 0 <= b_eat[t], b_eat[t] <= b_rating] #charging
    #constraints += [ 0 <= b_Q[t] ] #b_Q commented out

    #14 - Generator output limit
    constraints += [ 0 <= d_S[t], d_S[t] <= d_rating ]

#15 - Battery or generator does not discharge more than available
if len_t > 1:
    for t in range(1, len_t):
        constraints += [ b_gen[t]*dt <= j[t-1] ]
        constraints += [ d_S[t]*dt <= f[t-1] ]
else:
    constraints += [ b_gen[0]*dt <= j_start ]
    constraints += [ d_S[0]*dt <= f_start ]
