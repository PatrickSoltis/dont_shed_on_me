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

