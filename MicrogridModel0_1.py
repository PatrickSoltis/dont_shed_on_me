# %%​


# %%
# NOTE FOR EVERYONE
# TO MAKE CELLS, ADD THE FOLLOWING TEXT WITHOUT OQTES " # %% "
import numpy as np
import matplotlib.pyplot as plt
from cvxpy import *
import pandas as pd

# %%
print('Hello world')

# %%
# DATA IMPORT
office = pd.read_csv('SF_Office_import.csv') # Not planned in the microgrid
CARE = pd.read_csv('CARE_import.csv')
a10 = pd.read_csv('a10_import.csv')
e1 = pd.read_csv('e1_import.csv')
mbl = pd.read_csv('MBL_import.csv')
solar = pd.read_csv('solar_import.csv')

# %%
# Data vis
# Plot Office Load
plt.plot(office['Time (Hour)'],office['Power (kW)'], label = 'Office Load')
plt.title("Office Load Profile: 10 Day Time Horizon")
plt.xlabel("Time (Hours)")
plt.ylabel("Power (kW)")
plt.legend();


# %%
# Data Vis
# Plot MBL total and disagg load
plt.plot(mbl['Time (Hour)'],mbl['Total Power (kW)'], label="Total")
plt.plot(mbl['Time (Hour)'],mbl['Critical Power (kW)'], label="Microgrid Baseline")
plt.plot(mbl['Time (Hour)'],mbl['Remaining Power (kW)'], label="Remaining")
plt.title("MBL Load Profile: Disaggregated 10 Day Time Horizon")
plt.xlabel("Time (Hours)")
plt.ylabel("Power (kW)")
plt.legend();
# %%
# Data Vis
# Plot Resi total and disagg load
plt.plot(e1['Time (Hour)'],e1['Total Power (kW)'], label="Total")
plt.plot(e1['Time (Hour)'],e1['Critical Power (kW)'], label="Microgrid Baseline")
plt.plot(e1['Time (Hour)'],e1['Remaining Power (kW)'], label="Remaining")
plt.title("Residential Load Profile: Disaggregated 10 Day Time Horizon")
plt.xlabel("Time (Hours)")
plt.ylabel("Power (kW)")
plt.legend();
# %%
# Data Vis
# Plot Resi-CARE total and disagg load
plt.plot(CARE['Time (Hour)'],CARE['Total Power (kW)'], label="Total")
plt.plot(CARE['Time (Hour)'],CARE['Critical Power (kW)'], label="Microgrid Baseline")
plt.plot(CARE['Time (Hour)'],CARE['Remaining Power (kW)'], label="Remaining")
plt.title("Residential Load Profile: Disaggregated 10 Day Time Horizon")
plt.xlabel("Time (Hours)")
plt.ylabel("Power (kW)")
plt.legend();
# %%
# Data Vis
# Plot fire-station / hospital total and disagg load
plt.plot(a10['Time (Hour)'],a10['Total Power (kW)'], label="Total")
plt.plot(a10['Time (Hour)'],a10['Critical Power (kW)'], label="Microgrid Baseline")
plt.plot(a10['Time (Hour)'],a10['Remaining Power (kW)'], label="Remaining")
plt.title("Critical Facility Load Profile: Disaggregated 10 Day Time Horizon")
plt.xlabel("Time (Hours)")
plt.ylabel("Power (kW)")
plt.legend();

# %%
# Data Vis
# Plot Solar curves
plt.plot(solar['Time (Hour)'],solar['June (kW)'], label="June")
plt.plot(solar['Time (Hour)'],solar['September (kW)'], label="September")
plt.plot(solar['Time (Hour)'],solar['December (kW)'], label="December")
plt.title("Solar Gen: 10 Day Time Horizon")
plt.xlabel("Time (Hours)")
plt.ylabel("Power (kW)")
plt.legend();

# %%
# List of node indices
j_idx = np.arange(8)

### Define microgrid parameters ###

# Import nodal power demand values
D = pd.read_csv("nodes.csv")

#  Create apparent power demand dataframes
D_apparent_total = pd.DataFrame()
D_apparent_critical = pd.DataFrame()
D_apparent_remaining = pd.DataFrame()

#  Create active power demand dataframes
D_real_total = pd.DataFrame()
D_real_critical = pd.DataFrame()
D_real_remaining = pd.DataFrame()

#  Create reactive power demand dataframes
D_reactive_total = pd.DataFrame()
D_reactive_critical = pd.DataFrame()
D_reactive_remaining = pd.DataFrame()

# Power demand [MW] at each node
for x in j_idx:

    # Apparent power dataframes
    D_apparent_total["Node %1i"%x + " Total Apparent Power"] = D["Node %1i"%x + " (Apparent Power; Total)"]
    D_apparent_critical["Node %1i"%x + " Critical Apparent Power"] = D["Node %1i"%x + " (Apparent Power; Critical)"]
    D_apparent_remaining["Node %1i"%x + " Remaining Apparent Power"] = D["Node %1i"%x + " (Apparent Power; Remaining)"]

    # Real power dataframes
    D_real_total["Node %1i"%x + " Total Real Power"] = D["Node %1i"%x + " (Real Power; Total)"]
    D_real_critical["Node %1i"%x + " Critical Real Power"] = D["Node %1i"%x + " (Real Power; Critical)"]
    D_real_remaining["Node %1i"%x + " Remaining Real Power"] = D["Node %1i"%x + " (Real Power; Remaining)"]

    # Reactive power dataframes
    D_reactive_total["Node %1i"%x + " Total Reactive Power"] = D["Node %1i"%x + " (Reactive Power; Total)"]
    D_reactive_critical["Node %1i"%x + " Critical Reactive Power"] = D["Node %1i"%x + " (Reactive Power; Critical)"]
    D_reactive_remaining["Node %1i"%x + " Remaining Reactive Power"] = D["Node %1i"%x + " (Reactive Power; Remaining)"]

# Create solar power generated dataframe
s_s = pd.DataFrame()

# Define month of solar power to be examined
month = "June"

# Solar apparent power generated at each node (excludes power dispatched from batteries)
s_s["Node 0: " + month + " solar generation"] = 0
s_s["Node 1: " + month + " solar generation"] = solar[month + " (kW)"]
s_s["Node 2: " + month + " solar generation"] = solar[month + " (kW)"]

for x2 in np.arange(3,8):
    s_s["Node %1i: "%x2 + month + " solar generation"] = 0
s_s["Node 0: " + month + " solar generation"] = 0

# Solar real power generated at each node (excludes power dispatched from batteries)
s_p = s_s

# Solar reactive power generated at each node (excludes power dispatched from batteries)
s_q = s_s
s_q.iloc[:,:] = 0

# Maximum energy that batteries at nodes 1,2 and 7 can store (battery capacity)
j_max = np.array([0, 9.5, 9.5, 0, 0, 0, 0, 95])

# Maximum power discharge and charge rates
b_Srating = np.array([0, 25, 25, 0, 0, 0, 0, 40])

# Maximum power that diesel generator can produce (kW)
d_max = 20

# Priority ranking of different customer categories (5=highest priority, 1=lowest)
R = np.array([0, 0, 0, 0, 5, 4, 3, 2])

# Minimum nodal voltage
V_min = 0.95

# Maximum nodal voltage
V_max = 1.05

#Adding in dummy variables for r, x and I matrix until clarified:
# Resistance of each power line (between a node and its parent node)
r = np.array([0,0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])

# Reactance of each power line (between a node and its parent node)
x = np.array([0,0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01])

# Maximum current for each power line (between a node and its parent node)
I_max = np.array([0,1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2])

# Adjacency matrix where A[i, j]=1 if i is the parent of j
A = np.array([[1, 1, 1, 1, 1, 1, 1, 1],
              [0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0]])

# rho(j): Parent node of node j
rho = np.array([0, 0, 0, 0, 0, 0, 0, 0])

# %%
### Define optimization vars

    ### nodal power

# solar generation
#s is a parameter
S_p = Variable(8)
S_q = Variable(8)

# battery injection/absorption
b_S = Variable(8)
b_Q = Variable(8)
b_P = Variable(8)

# power delivered to load
l_S = Variable(8)
l_P = Variable(8)
l_Q = Variable(8)

# controllable generation (diesel)
d_S = Variable(8)
d_P = Variable(8)
d_Q = Variable(8)

# other
V = Variable(8) #voltage
F= Variable(8) #fraction of demand fulfilled
j= Variable(8) #battery state of charge

# power flow between nodes
P = Variable((8, 8))
Q = Variable((8, 8))
L = Variable((8,8))

# %%
### Define objective function
objective = Maximize(R.T@F)

# %%
### Define constraints. Numbering of constraints corresponds to report.
constraints = []

     ### Batteries ###

#1 - Battery energy availability
constraints += [ 0 <= j,
                 j <= j_max ]

#2 - Charge/discharge rating of battery
constraints += [ -b_Srating <= b_S,
                b_S <= b_Srating ]

#3 - Energy discharged <=energy available
constraints += [ b_S <=  j ] 

#7 - Battery can't store Q
constraints += [ 0 <= b_Q ]

    ### Diesel generator ###

#4 - Diesel generation capacity (power)
constraints += [ 0 <= d_S,
                d_S <= d_max ]

#10 - Fuel equation
#constraints += [ f[t]= f[t-1] - d_S * 0.08 ]

#11 - Cannot use more fuel than available
#constraints += [ d_S* 0.08 <= f[t] ]
 
    
    ### Other

#12 & 13 - Nodal voltage limits
constraints += [V_min**2 - V <= 0, 
                V - V_max**2 <= 0]

#14 - Squared line current limits
for kk in j_idx:

    constraints += [L[:,kk] - I_max**2 <= 0]

    ### Boundary conditions

# Power line flow for node 0
constraints += [P[0,0] == 0, Q[0,0] == 0]

# Squared line current for node 0
constraints += [L[0,0] == 0]

# Fix node 0 voltage to be 1 "per unit" (p.u.)
constraints += [V[0] == 1]

# Loop over each node
for jj in j_idx:
    
        ### Reactive/Active/Apparent power definitions at nodes ###
    constraints += [ norm(vstack([s_p.iloc[:,jj],s_q.iloc[:,jj]])) <= s_s.iloc[:,jj] ] #9 - solar p and q output tied to parameter input
    constraints += [ norm(vstack([b_P[jj],b_Q[jj]])) <= b_S[jj] ] #6 - batteries
    constraints += [ norm(vstack([l_P[jj],l_Q[jj]])) <= l_S[jj] ] #5 - demand
    constraints += [ norm(vstack([d_P[jj],d_Q[jj]])) <= d_S[jj] ] #8 - diesel generator

    # Parent node, i = \rho(j)
    i =  rho[jj]    

    # Line Power Flows- Constraint 21, this is wrong for our variables?
    #constraints += [0 == P[i, jj] - (l_P[jj] - s_p.iloc[0,jj]) - r[jj]*L[i, jj] - [A[jj,:]@P[jj,:]]]
    #Missing constraint 22-24, similar variable confusion


# %%
# Additional inequality constraints: 0jjmax

# Define constraints
constraints += [ b_S[0] == 0 ]#Constraint 15, no power stored - I think this needs modification to refer to the correct nodes.
constraints += [ b_S[3] == 0 ]
constraints += [ b_S[4] == 0 ]
constraints += [ b_S[5] == 0 ]
constraints += [ b_S[6] == 0 ]
constraints += [ F == l_S/D_real_total.iloc[0,:] ] #Constraint 16, Fraction of load served
constraints += [ l_S == s_s.iloc[0,:].values + b_S + d_S ] #Constraint 17a, total supply 
constraints += [ l_P == s_p.iloc[0, :].values + b_S + d_S ] #Constraint 17b, real power supply
constraints += [ l_Q == s_q.iloc[0, :].values + b_Q + d_Q ] #Constraint 17c, reactive power supply    
#j[t] = j[t-1] - b_S[t]*dt  #Constraint 18, time dimension of battery 
#j_0 = **choose initial value** A parameter?
constraints += [ l_S[4] == D_real_total.iloc[0,4] ]#Constraint 20, medical baseline #are we keeping this constraint?
constraints += [ l_S[5] == D_real_total.iloc[0,5] ] #Constraint 20, critical facility #are we keeping this constraint?

prob2 = Problem(objective, constraints)
prob2.solve()
print(prob2.status)
print("Program Results : %4.2f"%(prob2.value))
print(l_S.value)
print(l_P.value)
print(l_Q.value)