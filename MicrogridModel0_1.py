# %%


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
<<<<<<< HEAD
### Define optimization vars

    ### nodal power

# solar generation
#s is a parameter
p = Variable(7)
q = Variable(7)

# battery injection/absorption
b_S = Variable(7)
b_Q = Variable(7)
b_P = Variable(7)

# power delivered to load
l_S = Variable(7)
l_P = Variable(7)
l_Q = Variable(7)

# controllable generation (diesel)
d_S = Variable(7)
d_P = Variable(7)
d_Q = Variable(7)

# other
V = Variable(7) #voltage
F= Variable(7) #fraction of demand fulfilled
j= Variable(7) #battery state of charge

# power flow between nodes
P = Variable((7, 7))
Q = Variable((7, 7))
L = Variable((7, 7))

# %%
### Define objective function
objective = Maximize(R.T@F)

# %%
### Define constraints
constraints = []

    ### Reactive/Active/Apparent power definitions

constraints += [ norm(vstack([p[jj],q[jj]])) <= s[jj] ] #solar p and q output tied to parameter input
constraints += [ norm(vstack([b_P[jj],b_Q[jj]])) <= b_S[jj] ] #batteries
constraints += [ norm(vstack([l_P[jj],l_Q[jj]])) <= l_S[jj] ] #demand
constraints += [ norm(vstack([d_P[jj],d_Q[jj]])) <= d_S[jj] ] #diesel generator 

### Following constraints are verified but not yet sorted by Patrick
# Apparent Power Limits
constraints = [s - s_max <= 0, -p <= 0, -q <= 0, norm(vstack([sum(p), sum(q)]))- sum(s)<= 0]
#Not sure about this last one

# Nodal voltage limits
constraints += [v_min**2 - V <= 0, V - v_max**2 <= 0]

# Squared line current limits
constraints += [L - I_max**2 <= 0]

# Boundary condition for power line flows
constraints += [P[0,0] == 0, Q[0,0] == 0]

# Boundary condition for squared line current
constraints += [L[0,0] == 0]

# Fix node 0 voltage to be 1 "per unit" (p.u.)
constraints += [V[0] == 1]

# Loop over each node
for jj in j_idx:
    
    # Parent node, i = \rho(j)
    i =  rho[jj]    
    
    # Line Power Flows
    constraints += [P[i, jj] - (l_P[jj] - p[jj]) - r[i, jj]*L[i, jj] - sum([A[

# %%
# Additional inequality constraints: 0jjmax

0 <= j <= jmax
-b_Srating <= b_S[b] <= b_Srating
b_S*dt <=  jt-1 
0 <= b_S[d] <= dmax

# Define constraints
b_S[n] = 0
F = l_S/D
l_S = s + b_S        
j[t] = j[t-1] - b_S[t]*dt   
#j_0 = **choose initial value** A parameter?
l_Sk = Dk

0 <= b_Q 

=======

# Jack's section

# List of node indices
i_idx = np.arange(8)

## Define microgrid parameters ##

# import nodal power demand values
D = pd.read_csv("nodes.csv")
D_apparent_total = pd.DataFrame()
D_apparent_critical = pd.DataFrame()
D_apparent_remaining = pd.DataFrame()

# Apparent power demand [MW] at each node
for x in i_idx:
    D_apparent_total["Node %1i"%x + " Total Apparent Power"] = D["Node %1i"%x + " (Apparent Power; Total)"]
    D_apparent_critical["Node %1i"%x + " Critical Apparent Power"] = D["Node %1i"%x + " (Apparent Power; Critical)"]
    D_apparent_remaining["Node %1i"%x + " Remaining Apparent Power"] = D["Node %1i"%x + " (Apparent Power; Remaining)"]

# Apparent power generated at each node (excludes power dispatched from batteries)
s = np.array([0, 0, 0, 0, 0, 0, 0, 0])

# Active power generated at each node (excludes power dispatched from batteries)
p = np.array([0, 0, 0, 0, 0, 0, 0, 0])

# Reactive power generated at each node (excludes power dispatched from batteries)
q = np.array([0, 0, 0, 0, 0, 0, 0, 0])

# Maximum energy that batteries at nodes 1,2 and 7 can store (battery capacity)
j_max = np.array([0, 0, 0])

# Maximum power discharge and charge rates
# Does this only apply to nodes 1,2 and 7?
#bS_rating = np.zeros((2,3), dtype=int)

# Maximum power that diesel generator can produce
d_max = 0

# Priority ranking of different customer categories (5=highest priority, 1=lowest)
R = np.array([5, 4, 3, 2])

# Power factor (i.e., assumed ratio of active power to apparent power)
pf = np.array([0, 0, 0, 0, 0, 0, 0, 0])

# Minimum nodal voltage
V_min = 0

# Maximum nodal voltage
V_max = 0

# Resistance of each power line (between a node and its parent node)
r = np.array([0, 0, 0, 0, 0, 0, 0])

# Reactance of each power line (between a node and its parent node)
x = np.array([0, 0, 0, 0, 0, 0, 0])

# Maximum current for each power line (between a node and its parent node)
I_max = np.array([0, 0, 0, 0, 0, 0, 0])

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
>>>>>>> b25d4f0f2d4bfc69a6785ca993b464da03dde3db
