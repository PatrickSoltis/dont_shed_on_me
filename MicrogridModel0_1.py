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
office = pd.read_csv('SF_Office_import.csv')
CARE = pd.read_csv('CARE_import.csv')
a1 = pd.read_csv('a1_import.csv')
e1 = pd.read_csv('e1_import.csv')
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
# Plot other loads
plt.plot(CARE['Time (Hour)'],CARE['Power (kW)'], label="Res. CARE")
plt.plot(e1['Time (hour)'],e1['Power (kW)'], label="Res.")
plt.plot(a1['Time (Hour)'],a1['Power (kW)'], label="Non-Res CF")
plt.title("CARE, Resi, Crit Load Load Profiles: 10 Day Time Horizon")
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
