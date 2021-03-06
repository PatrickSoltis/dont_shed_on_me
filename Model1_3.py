#%%
import numpy as np
import matplotlib.pyplot as pyplot
from cvxpy import *
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('ggplot')

#%% DATA IMPORTS & SELECTION
#Apparent power values assume pf=0.95

#Source data available for viewing (not used in optimization)
office = pd.read_csv('SF_Office_import.csv') #Office building - not used in model
CARE = pd.read_csv('CARE_import.csv') #CARE/FERA residential
a10 = pd.read_csv('a10_import.csv') #Fire station
e1 = pd.read_csv('e1_import.csv') #Residential
mbl = pd.read_csv('MBL_import.csv') #Medical baseline ???
solar = pd.read_csv('solar_import.csv')

#Aggregated load profiles
D_df = pd.read_csv('nodes.csv')

#Select hour numbers to use from load and solar data
# Scenarios: 55 hours, 5 days, 10 days
t0 = 0
len_t = 120

#Select month of solar data (June/September/December)
month = 'June'

#%% PARAMETERS

#1 - Value ranking of different customer categories (5=highest priority, 1=lowest)
R = np.array([[0, 0, 0, 0, 5, 4, 3, 2]])
#   R array is 2D, but remaining length-8 arrays are only 1 dimension.

#2 - Load total power for each node into D
D = D_df.iloc[t0:t0+len_t, [1,10,19,28,37,46,55,64]].to_numpy()
D_P = D_df.iloc[t0:t0+len_t, [4,13,22,31,40,49,58,67]].to_numpy()
D_Q = D_df.iloc[t0:t0+len_t, [7,16,25,34,43,52,61,70]].to_numpy()
#   Index as: D[hour number, node number]

#3 - solar PV generation
hourly_gen = np.array(solar[month+' (kW)'].values[t0:t0+len_t])
zeros = np.zeros(len_t)
S_S = np.array([zeros, hourly_gen, hourly_gen, zeros, zeros, zeros, zeros, zeros]).T
#   Index as: S_S[hour number, node number]

#4 - battery energy (batteries at nodes 1 & 2, EV at 7)
j_max = np.array([0, 10, 10, 0, 0, 0, 0, 100]) 
j_start = np.array([0, 5, 5, 0, 0, 0, 0, 50])# Chosen to start at 50% charge

#5 - diesel fuel (diesel generator at node 3)
# Scenarios: 80 kWh, 320 kWh, Unlimited kWh (set to at least demand value)
f_start = np.array([0, 0, 0, 10000, 0, 0, 0, 0]) 

#6 - power ratings
#   10kWh batteries at nodes 1 & 2, 100 kWh battery at node 7. Can only discharge 95% of capacity.
b_rating = np.array([0, 9.5, 9.5, 0, 0, 0, 0, 95]) 
#   One 20kW diesel generator at node 3
d_rating = np.array([0, 0, 0, 20, 0, 0, 0, 0]) 

#7 - power factors (currently is implicit 0.95 in D_df columns)
#pf = np.full((1,8), 0.95)

#8 - voltage limits
V_min = 0.5 #0.95
V_max = 3 #1.05 #

#9 - Resistance and reactance
r = np.array([0,0.07, 0.07, 0.07, 0.07, 0.07, 0.07, 0.07])
x = np.array([0,0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02])

#10 - Max current
I_max = np.array([0,5, 5, 5, 5, 5, 5, 5])

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
#Node 0 has no parent

#12 - Node indexing
j_idx = np.arange(8) #Index including all 8 nodes
rho = np.array([0, 0, 0, 0, 0, 0, 0, 0]) #Defines parent nodes in for loops

#13 - Time step
dt = 1

#14 - Inverter efficiencies
nu_s = 1 #solar
nu_b = 1 #battery
#   These values are not yet built into constraints. 
#   No efficiency value for generator because we assume generator energy already accounts for efficiency losses.

#%% OPTIMIZATION VARIABLES
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
S_P = Variable((len_t,8))
S_Q = Variable((len_t,8))
#If solar provides only real power, add to constraints: S_P = S_S

#6 - Net power
s = Variable((len_t,8))
p = Variable((len_t,8))
q = Variable((len_t,8))

#7 - Voltage
V = Variable((len_t,8))

#8 - Energy stock
j = Variable((len_t,8)) #Battery state of charge
f = Variable((len_t,8)) #Diesel fuel available

#9-11 - Line variables
P = Variable((len_t,8)) #active power flow
Q = Variable((len_t,8)) #reactive power flow
L = Variable((len_t,8)) #squared magnitude of complex current

#%% OBJECTIVE FUNCTION

objective = Minimize( -sum(F@R.T) )

#%% CONSTRAINTS

#1 - Define fraction of load served (may need to loop?)
constraints = [ F == l_P/D_P ]

#2 - Node 0 has no line connecting itself, and reference voltage
constraints += [ P[:,0] == 0, Q[:,0] == 0, L[:,0] == 0]
constraints += [ V[:,0] == 1 ]

#3 - Net power consumed at each node
constraints += [ s == l_S - b_S - d_S - S_S ]
constraints += [ p == l_P - b_P - d_P - S_P ]
constraints += [ q == l_Q - b_Q - d_Q - S_Q ]

#4 - No phantom batteries, generators, or PV
no_batteries = [0, 3, 4, 5, 6]
no_generator = [0, 1, 2, 4, 5, 6, 7]
no_solar = [0, 3, 4, 5, 6, 7]

constraints += [ b_S[:, no_batteries] == 0 ]
constraints += [ b_P[:, no_batteries] == 0 ]
constraints += [ b_Q[:, no_batteries] == 0 ]

constraints += [ d_S[:, no_generator] == 0 ]
constraints += [ d_P[:, no_generator] == 0 ]
constraints += [ d_Q[:, no_generator] == 0 ]

constraints += [ S_P[:, no_solar] == 0 ]
constraints += [ S_Q[:, no_solar] == 0 ]
    
#5 - Power delivered cannot exceed demand
constraints += [ l_P <= D_P ]

#6 - Battery state of charge
constraints += [ j[0] == j_start ]
if len_t > 1:
    for t in range(1,len_t):
        constraints += [ j[t] == j[t-1] - b_S[t-1]*dt]
for t in range(len_t):
    constraints += [ 0 <= j[t], j[t] <= j_max]

#7 - Diesel generator fuel stock
constraints += [ f[0] == f_start ]
if len_t > 1:
    for t in range(1, len_t):
        constraints += [ f[t] == f[t-1] - d_S[t-1]*dt ]
constraints += [ 0 <= f ]

#8 - Battery or generator does not discharge more than available
constraints += [ b_S[0]*dt <= j_start ]
constraints += [ d_S[0]*dt <= f_start ]
if len_t > 1:
    for t in range(1, len_t):
        constraints += [ b_S[t]*dt <= j[t-1] ]
        constraints += [ d_S[t]*dt <= f[t-1] ]

for t in range(len_t):

    #9 - Battery (dis)charging limit
    constraints += [ -b_rating <= b_S[t], b_S[t] <= b_rating ]
    constraints += [ 0 <= b_Q[t] ] # ???

    #10 - Generator output limit
    constraints += [ d_S[t] <= d_rating ]

    for jj in j_idx:
        i = rho[jj]

        #11 - DistFlow equations
        constraints += [ P[t,jj] == p[t,jj] + r[jj]*L[t,jj] + A[jj]@P[t,:] ] 
        constraints += [ Q[t,jj] == q[t,jj] + x[jj]*L[t,jj] + A[jj]@Q[t,:] ]

        #12 - Voltage drop
        constraints += [ V[t,jj] - V[t,i] == (r[jj]**2 + x[jj]**2)*L[t,jj] - 2*(r[jj]*P[t,jj].T + x[jj]*Q[t,jj].T) ]
    
        #13 - Definition of squared magnitude of complex current
        constraints += [ quad_over_lin(vstack([P[t,jj],Q[t,jj]]), V[t,jj]) <= L[t,jj] ]

        #14 - Definitions of apparent power consumption and production (for solar, battery, and diesel)
        constraints += [ norm(vstack([l_P[t,jj],l_Q[t,jj]])) <= l_S[t,jj] ]
        constraints += [ norm(vstack([S_P[t,jj],S_Q[t,jj]])) <= S_S[t,jj] ]
        constraints += [ norm(vstack([b_P[t,jj],b_Q[t,jj]])) <= b_S[t,jj] ]
        constraints += [ norm(vstack([d_P[t,jj],d_Q[t,jj]])) <= d_S[t,jj] ]

    #15 - Definitions of total apparent power across system
    #RECOMMENDED removal - Optimization gives same results without
    #constraints += [ norm(vstack([sum(l_P[t,:]),sum(l_Q[t,:])])) <= sum(l_S[t,:]) ]
    #constraints += [ norm(vstack([sum(S_P[t,:]),sum(S_Q[t,:])])) <= sum(S_S[t,:]) ]
    #constraints += [ norm(vstack([sum(b_P[t,:]),sum(b_Q[t,:])])) <= sum(b_S[t,:]) ]
    #constraints += [ norm(vstack([sum(d_P[t,:]),sum(d_Q[t,:])])) <= sum(d_S[t,:]) ]

    #16 - Power supplied (l) cannot exceed power generated
    constraints += [ sum(l_S[t, :]) <= sum(b_S[t, :] + d_S[t, :] + S_S[t, :]) ]
    constraints += [ sum(l_P[t, :]) <= sum(b_P[t, :] + d_P[t, :] + S_P[t, :]) ]
    constraints += [ sum(l_Q[t, :]) <= sum(b_Q[t, :] + d_Q[t, :] + S_Q[t, :]) ] 

    #17 - Allowed voltages
    constraints += [ V_min**2 <= V[t], V[t] <= V_max**2 ]

    #18 - Current limit
    constraints += [ L[t] <= I_max**2 ]

#19 - Power consumed/generated cannot be negative
constraints += [d_S >= 0, d_P >= 0, 
                l_S >= 0, l_P >= 0, 
                S_P >= 0 ]

#%% SOLVE
prob = Problem(objective, constraints)
prob.solve()
print(prob.status)
print(prob.value)

#%% FRACTIONS SERVED
for t in range(len_t):
    print("Time %3.0f"%(t))
    for jj in j_idx:
        print("Node %2.0f fraction served: %1.3f"%(jj, F[t,jj].value))


# %% Look at & Plot Underlying Demand/Supply/Generation Values at Each Node

# Define function that creates a data frame of variables or arrays from output
def var_df(var):
    if type(var) == cvxpy.expressions.variable.Variable: 
        new_df = pd.DataFrame(var.value).round(3)
    else: 
        new_df = pd.DataFrame(var).round(3)
    total = pd.DataFrame(new_df.sum(axis=1), columns = ['total']).round(3)
    new_df = new_df.join(total)
#     avg = new_df.mean()#.to_numpy().transpose()
#     new_df = new_df.append(avg, ignore_index=True)
    return new_df


# Add column to scenarios dataframe for scenario outputs
F_df = var_df(F)
arrays = [[str(len_t)], [str(f_start[3])]]
tuples = list(zip(*arrays))
index = pd.MultiIndex.from_tuples(tuples, names = ['Outage Duration (hrs)', 'Diesel Fuel (kWh)'])
scenarios = pd.DataFrame(data = F_df.mean()[4:8], columns = index).rename(index = lambda x: 'Node '+str(x))#.reindex(nodes)
scenarios


#  %% Plot fraction of load served over time by node
lines = ['solid', 'dashed', 'dashdot', 'dotted']
sizes = [1, 1, 1, 2]
colors = ['c', 'tomato', 'deeppink', 'purple']
for i in np.arange(4, 8): 
    plt.plot(F_df.index, F_df[i], linestyle = lines[i-4], label = ('Node '+str(i)), 
             linewidth = sizes[i-4], c = colors[i-4])
plt.plot(F_df.index, F_df.mean(axis = 1), label = 'Mean', c = 'dimgrey', linewidth = 1)
#plt.title('Fraction of Load Served by Node', fontsize = 16)
scenario = str(len_t)+' hr Outage, '+str(f_start[3])+' kWh diesel fuel'
plt.title(scenario, fontsize = 16)
plt.ylabel('Fraction Served', fontsize = 14)
plt.xlabel('Hour', fontsize = 14)
plt.legend(bbox_to_anchor = (1, 1), loc = 'upper left', title = "Legend", title_fontsize = 12)
#plt.legend(loc = 'center left')
plt.show()

print('Average fraction of load served by node:')
print('Scenario:', len_t, 'hrs,', f_start[3], 'kWh fuel')
F_df.mean()[4:8]


# %% Create lists of active, reactive, and apparent P variables/arrays
active_P_vars = [D_P, l_P, b_P, S_P, d_P]#p
reactive_P_vars = [D_Q, l_Q, b_Q, S_Q, d_Q]#q
apparent_P_vars = [D, l_S, b_S, S_S, d_S]#s
active_P_df_list = [1, 2, 3, 4, 5] 
reactive_P_df_list = [1, 2, 3, 4, 5] #'Demand', 'Supply', 'Solar', 'Battery', 'Diesel', 'Net Power'
apparent_P_df_list = [1, 2, 3, 4, 5] #'Demand', 'Supply', 'Solar', 'Battery', 'Diesel', 'Net Power'
names = ['Demand', 'Supply', 'Battery', 'Solar', 'Diesel']
colors = ['indigo', 'indigo', 'limegreen', 'gold', 'mediumvioletred']
lines = ['dotted', 'solid', 'dashed', 'dashed', 'dashed']
lwidth = [2, 3, 2, 2, 2]

# Create lists of active, reactive, and apparent power dataframes
for i in np.arange(len(active_P_vars)):
    active_P_df_list[i] = var_df(active_P_vars[i])
for i in np.arange(len(reactive_P_vars)):
    reactive_P_df_list[i] = var_df(reactive_P_vars[i])
for i in np.arange(len(apparent_P_vars)):
    apparent_P_df_list[i] = var_df(apparent_P_vars[i])

# Plot Active Power Demand, Supply, & Generation Over Time
def plots_dfs(df_list, title, ylabel):
    for i in np.arange(len(df_list)):
        plt.plot(df_list[i].index, df_list[i]['total'], 
                 label = names[i], linestyle = lines[i], c = colors[i], linewidth = lwidth[i])
    plt.title(title, fontsize = 16)
    plt.ylabel(ylabel, fontsize = 14)
    plt.xlabel('Hour', fontsize = 14)
    plt.legend(bbox_to_anchor = (1, 1), loc = 'upper left', title = "Legend", title_fontsize = 12)
    #plt.legend()
    plt.show()
    
plots_dfs(active_P_df_list, scenario, 'Active Power [kW]')
plots_dfs(reactive_P_df_list, scenario, 'Reactiver Power [kVAR]')
plots_dfs(apparent_P_df_list, scenario, 'Apparent Power [kVA]')
# plots_dfs(active_P_df_list, 'Active Power Supply, Generation, & Demand', 'Active Power [kW]')
# plots_dfs(reactive_P_df_list, 'Reactive Power Supply, Generation, & Demand', 'Reactiver Power [kVAR]')
# plots_dfs(apparent_P_df_list, 'Apparent Power Supply, Generation, & Demand', 'Apparent Power [kVA]')


# %%
