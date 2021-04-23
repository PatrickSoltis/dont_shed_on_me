# %%


# %%
# NOTE FOR EVERYONE
# TO MAKE CELLS, ADD THE FOLLOWING TEXT WITHOUT OQTES " # %% "
import numpy as np
import matplotlib.pyplot as plt
from cvxpy import *
import pandas as pd

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

# Jack's section

# List of node indices
i_idx = np.arange(8)

## Define microgrid parameters ##

# Import nodal power demand values
D = pd.read_csv("nodes.csv")

#  Create apparent power dataframes
D_apparent_total = pd.DataFrame()
D_apparent_critical = pd.DataFrame()
D_apparent_remaining = pd.DataFrame()

#  Create active power dataframes
D_active_total = pd.DataFrame()
D_active_critical = pd.DataFrame()
D_active_remaining = pd.DataFrame()

#  Create reactive power dataframes
D_reactive_total = pd.DataFrame()
D_reactive_critical = pd.DataFrame()
D_reactive_remaining = pd.DataFrame()

# Power demand [MW] at each node
for x in i_idx:

    # Apparent power dataframes
    D_apparent_total["Node %1i"%x + " Total Apparent Power"] = D["Node %1i"%x + " (Apparent Power; Total)"]
    D_apparent_critical["Node %1i"%x + " Critical Apparent Power"] = D["Node %1i"%x + " (Apparent Power; Critical)"]
    D_apparent_remaining["Node %1i"%x + " Remaining Apparent Power"] = D["Node %1i"%x + " (Apparent Power; Remaining)"]

    # Active power dataframes
    D_active_total["Node %1i"%x + " Total Active Power"] = D["Node %1i"%x + " (Active Power; Total)"]
    D_active_critical["Node %1i"%x + " Critical Active Power"] = D["Node %1i"%x + " (Active Power; Critical)"]
    D_active_remaining["Node %1i"%x + " Remaining Active Power"] = D["Node %1i"%x + " (Active Power; Remaining)"]

    # Reactive power dataframes
    D_reactive_total["Node %1i"%x + " Total Reactive Power"] = D["Node %1i"%x + " (Reactive Power; Total)"]
    D_reactive_critical["Node %1i"%x + " Critical Reactive Power"] = D["Node %1i"%x + " (Reactive Power; Critical)"]
    D_reactive_remaining["Node %1i"%x + " Remaining Reactive Power"] = D["Node %1i"%x + " (Reactive Power; Remaining)"]

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
# bS_rating = np.zeros((2,3), dtype=int)

# Maximum power that diesel generator can produce
d_max = 0

# Priority ranking of different customer categories (5=highest priority, 1=lowest)
R = np.array([5, 4, 3, 2])

# Power factor (i.e., assumed ratio of active power to apparent power)
# pf = np.array([0, 0, 0, 0, 0, 0, 0, 0])

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

# %%
