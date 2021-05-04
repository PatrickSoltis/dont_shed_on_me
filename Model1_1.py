#%%
import numpy as np
import matplotlib.pyplot as pyplot
from cvxpy import *
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('ggplot')

#%% Data Imports

#Apparent power values assume pf=0.95
office = pd.read_csv('SF_Office_import.csv') #Office building - not used in model
CARE = pd.read_csv('CARE_import.csv') #CARE/FERA residential
a10 = pd.read_csv('a10_import.csv') #Fire station
e1 = pd.read_csv('e1_import.csv') #Residential
mbl = pd.read_csv('MBL_import.csv') #Medical baseline ???
solar = pd.read_csv('solar_import.csv')

#%% Parameters

#Parameters A - Direct Objective Inputs (1-2)

D_df = pd.read_csv('nodes.csv')

#1 - Value ranking of different customer categories (5=highest priority, 1=lowest)
R = np.array([[0, 0, 0, 0, 5, 4, 3, 2]])
#Remaining length-8 arrays are only 1 dimension. 
# They will only be used in constraints at a single time.

#Select hour numbers to use from load and solar data
t0 = 12
len_t = 85

#2 - Load total power for each node into D
D = D_df.iloc[t0:t0+len_t, [1,10,19,28,37,46,55,64]].to_numpy()
D_P = D_df.iloc[t0:t0+len_t, [4,13,22,31,40,49,58,67]].to_numpy()
D_Q = D_df.iloc[t0:t0+len_t, [7,16,25,34,43,52,61,70]].to_numpy()
#Index as: D[hour number, node number]

#Parameters B (3-8)

#3 - solar PV generation
month = 'June' #select month of solar data (June/September/December)
hourly_gen = np.array(solar[month+' (kW)'].values[t0:t0+len_t])
zeros = np.zeros(len_t)
s_S = np.array([zeros, hourly_gen, hourly_gen, zeros, zeros, zeros, zeros, zeros]).T
#Index as: s_S[hour number, node number]

#4 - battery energy (batteries at nodes 1 & 2, EV at 7)
j_max = np.array([0, 9.5, 9.5, 0, 0, 0, 0, 95])
j_start = np.array([0, 4.0, 4.0, 0, 0, 0, 0, 30])#arbitrarily chosen values

#5 - diesel fuel (diesel generator at node 3)
f_start = np.array([0, 0, 0, 10, 0, 0, 0, 0]) #arbitrarily chosen values

#6 - power ratings
b_rating = np.array([0, 8, 8, 0, 0, 0, 0, 20]) #battery
d_rating = np.array([0, 0, 0, 10, 0, 0, 0, 0])

#7 - power factors (currently is implicit 0.95 in D_df columns)
#pf = np.full((1,8), 0.95)

#8 - voltage limits
V_min = 0.5 #0.95
V_max = 10 #1.05 #

# Check available energy
print('D:',np.sum(D))
print('s_S:',np.sum(s_S))
print('j_start:',np.sum(j_start))
print('f_start:',np.sum(f_start))
print('energy available:',np.sum([np.sum(s_S),np.sum(j_start),np.sum(f_start)]))

#Parameters C (9-11)

j_idx = np.arange(8)

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

#Rho - Defines parent nodes in for loops
rho = np.array([0, 0, 0, 0, 0, 0, 0, 0])

#Parameters D (12-)

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

# %% Define objective function

#objective = Maximize( sum(F@R.T) )
objective = Minimize( -sum(F@R.T) )

# %% Constraints

# Constraints 0

#Nothing between node 0 and itself
constraints = [ P[:,0] == 0, Q[:,0] == 0, L[:,0] == 0]
#Fix node 0 voltage to be 1 p.u.
constraints += [ V[:,0] == 1 ]

# Constraints A (1-4)

#1 - Net power consumed at each node
constraints = [ p == l_P - b_P - d_P - s_P ]
constraints += [ q == l_Q - b_Q - d_Q - s_Q ]
constraints += [ s == l_S - b_S - d_S - s_S ]

#2 - No phantom batteries or generators
no_batteries = [0, 3, 4, 5, 6]
no_generator = [0, 1, 2, 4, 5, 6, 7]
no_solar = [0, 3, 4, 5, 6, 7]

for node in no_batteries:
    constraints += [ b_S[:, node] == 0 ]
    constraints += [ b_P[:, node] == 0 ]
    constraints += [ b_Q[:, node] == 0 ]
for node in no_generator:
    constraints += [ d_S[:, node] == 0 ]
    constraints += [ d_P[:, node] == 0 ]
    constraints += [ d_Q[:, node] == 0 ]
for node in no_solar:
    constraints += [ s_P[:, node] == 0 ]
    constraints += [ s_Q[:, node] == 0 ]
    
#3 - Define fraction of load served (may need to loop?)
#constraints += [ F == l_S/D ]
constraints += [ F == l_P/D_P ]

#4 - Guarantee full load for critical nodes
# critical = [4, 5]
# for node in critical:
#     constraints += [ F[:, node] == 1. ]

#4.x - Power delivered cannot exceed demand
#constraints += [ l_S <= D ]
constraints += [ l_P <= D_P ]
#constraints += [ l_Q <= D_Q ]

#Need to limit apparent power. In Homework, it was done this way:
#constraints = [ s <= s_max ]
#where s is apparent power generated at each node, and s was generator limit

# Constraints B (5-6)

#5 - Battery state of charge
constraints += [ j[0] == j_start ]
if len_t > 1:
    for t in range(1,len_t):
        constraints += [ j[t] == j[t-1] - b_S[t-1]*dt]
for t in range(len_t):
    constraints += [ 0 <= j[t], j[t] <= j_max]

#6 - Fuel stock
constraints += [ f[0] == f_start ]
if len_t > 1:
    for t in range(1, len_t):
        constraints += [ f[t] == f[t-1] - d_S[t-1]*dt ]

# Constraints C (7-9, 14)

for t in range(len_t):
    for jj in j_idx:
        i = rho[jj]

        #7 - DistFlow equations
        constraints += [ P[t,jj] == p[t,jj] + r[jj]*L[t,jj] + A[jj]@P[t,:] ] 
        constraints += [ Q[t,jj] == q[t,jj] + x[jj]*L[t,jj] + A[jj]@Q[t,:] ]

#         #8 - Voltage drop
        constraints += [ V[t,jj] - V[t,i] == (r[jj]**2 + x[jj]**2)*L[t,jj] - 2*(r[jj]*P[t,jj].T + x[jj]*Q[t,jj].T) ]

#         #9 - Squared current magnitude (relaxed)
        constraints += [ quad_over_lin(vstack([P[t,jj],Q[t,jj]]), V[t,jj]) <= L[t,jj] ]

#         #14 - Definition of apparent power
#        constraints += [ norm(vstack([p[t,jj],q[t,jj]])) <= s[t,jj] ]
        # Homework only checked this relationship generation, this is for net power
        # MB note: instead of applying apparent power constraint for cumulative, apply to each source (s/b/d)
        
#         # New - Definition of apparent power for generation
        constraints += [ norm(vstack([l_P[t,jj],l_Q[t,jj]])) <= l_S[t,jj] ]
        constraints += [ norm(vstack([s_P[t,jj],s_Q[t,jj]])) <= s_S[t,jj] ]
        constraints += [ norm(vstack([b_P[t,jj],b_Q[t,jj]])) <= b_S[t,jj] ]
        constraints += [ norm(vstack([d_P[t,jj],d_Q[t,jj]])) <= d_S[t,jj] ]
    constraints += [ norm(vstack([sum(l_P[t,:]),sum(l_Q[t,:])])) <= sum(l_S[t,:]) ]
    constraints += [ norm(vstack([sum(s_P[t,:]),sum(s_Q[t,:])])) <= sum(s_S[t,:]) ]
    constraints += [ norm(vstack([sum(b_P[t,:]),sum(b_Q[t,:])])) <= sum(b_S[t,:]) ]
    constraints += [ norm(vstack([sum(d_P[t,:]),sum(d_Q[t,:])])) <= sum(d_S[t,:]) ]
    
    # Ensure that power supplied l is less than or equal to power generated
    constraints += [sum(l_S[t, :]) <= sum(b_S[t, :] + d_S[t, :] + s_S[t, :]), 
                    sum(l_P[t, :]) <= sum(b_P[t, :] + d_P[t, :] + s_P[t, :]), 
                    sum(l_Q[t, :]) <= sum(b_Q[t, :] + d_Q[t, :] + s_Q[t, :])] 

# Constraints E (11-13)

for t in range(len_t):
    
    #11 - Battery (dis)charging limit
    constraints += [ -b_rating <= b_S[t], b_S[t] <= b_rating ]
    constraints += [ 0 <= b_Q[t] ]

    #12 - Generator output limit
    constraints += [ d_S[t] <= d_rating ]

#13 - Battery or generator does not discharge more than available
constraints += [ b_S[0]*dt <= j_start ]
constraints += [ d_S[0]*dt <= f_start ]
if len_t > 1:
    for t in range(1, len_t):
        constraints += [ b_S[t]*dt <= j[t-1] ]
        constraints += [ d_S[t]*dt <= f[t-1] ]

# Constraints F (15-16)

for t in range(len_t):

    #15 - Allowed voltages
    constraints += [ V_min**2 <= V[t], V[t] <= V_max**2 ]

#16 - Current limit
constraints += [ L[t] <= I_max**2 ]

# Constraints G
# Ensure that power consumed/generated is greater than or equal to 0
constraints += [d_S >= 0,
                d_P >= 0, 
                l_S >= 0, 
                l_P >= 0, 
                s_P >= 0]

# %% Solve
prob = Problem(objective, constraints)
prob.solve()
print(prob.status)
print(prob.value)

for t in range(len_t):
    print("Time %3.0f"%(t))
    for jj in j_idx:
        print("Node %2.0f fraction served: %1.3f"%(jj, F[t,jj].value))

# %% Define function that creates a data frame of variables or arrays from output
def var_df(var):
    if type(var) == cvxpy.expressions.variable.Variable: 
        new_df = pd.DataFrame(var.value).round(3)
    else: 
        new_df = pd.DataFrame(var).round(3)
    total = pd.DataFrame(new_df.sum(axis=1), columns = ['total']).round(3)
    new_df = new_df.join(total)
    return new_df

# Plot fraction of load served over time by node
F_df = var_df(F)
lines = ['solid', 'dashed', 'dashdot', 'dotted']
sizes = [1, 1, 1, 2]
colors = ['c', 'tomato', 'deeppink', 'purple']
for i in np.arange(4, 8): 
    plt.plot(F_df.index, F_df[i], linestyle = lines[i-4], label = ('Node '+str(i)), 
             linewidth = sizes[i-4], c = colors[i-4])
plt.title('Fraction of Load Served by Node', fontsize = 16)
plt.ylabel('Fraction Served', fontsize = 14)
plt.xlabel('Hour', fontsize = 14)
plt.legend()
plt.show()

# %% Create lists of active, reactive, and apparent P variables/arrays
active_P_vars = [D_P, l_P, b_P, s_P, d_P]#p
reactive_P_vars = [D_Q, l_Q, b_Q, s_Q, d_Q]#q
apparent_P_vars = [D, l_S, b_S, s_S, d_S]#s
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
    plt.legend()
    plt.show()
    
plots_dfs(active_P_df_list, 'Active Power Supply, Generation, & Demand', 'Active Power [kW]')
plots_dfs(reactive_P_df_list, 'Reactive Power Supply, Generation, & Demand', 'Reactiver Power [kVAR]')
plots_dfs(apparent_P_df_list, 'Apparent Power Supply, Generation, & Demand', 'Apparent Power [kVA]')

# %% Define function to take negative and positive data apart and cumulate
def get_cumulated_array(data, **kwargs):
    cum_arr = data.clip(**kwargs)
    cum_arr = np.cumsum(cum_arr, axis=0)
    d = np.zeros(np.shape(data))
    d[1:] = cum_arr[:-1]
    return d  

# Define function to create plots
def plots(supp, batt, solar, diesel, title, ylabel):
    data = np.array([-supp, batt, solar, diesel])
    data_shape = np.shape(data)
    cumulated_data = get_cumulated_array(data, min=0)
    cumulated_data_neg = get_cumulated_array(data, max=0)
    row_mask = (data<0)
    cumulated_data[row_mask] = cumulated_data_neg[row_mask]
    data_stack = cumulated_data
    labels = ["Power Supplied", "Battery", "Solar PV", "Diesel Generator"]
    names = ['Demand', 'Supply', 'Battery', 'Solar', 'Diesel']
    colors = ['indigo', 'limegreen', 'gold', 'mediumvioletred']
    fig = plt.figure()
    ax = plt.subplot(111)
    for i in np.arange(0, data_shape[0]):
        ax.bar(np.arange(data_shape[1]), data[i], bottom=data_stack[i], label=labels[i],
               color = colors[i])
    plt.axhline(color = 'black', linewidth = 0.5)
    plt.legend()
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel('Node')
    maxes = [np.max(cumulated_data), np.max(data)]
    plt.ylim(top = np.max(maxes)+2)
    plt.show()

# Create dataframe of outputs
t = 0
outputs = pd.DataFrame({'Power':P.value[t], 'net power': s.value[t], 'net active p': p.value[t], 'net reactive p': q.value[t], 
                        'App Demand':D[t],'App Supply':l_S.value[t], 'App Battery':b_S.value[t], 
                        'App Solar': s_S[t],'App Diesel':d_S.value[t], 
                        'Active Demand':D_P[t], 'Active Supply': l_P.value[t], 'Active Battery': b_P.value[t], 
                        'Active Solar':s_P.value[t], 'Active Diesel': d_P.value[t], 
                        'Reactive Demand': D_Q[t], 'Reactive Supply': l_Q.value[t], 'Reactive Battery':b_Q.value[t], 
                        'Reactive Solar': s_Q.value[t], 'Reactive Diesel': d_Q.value[t], 'Voltage': V.value[t]}, 
                         dtype = float)
outputs = outputs.round(2)
outputs

# %% Plot outputs
# Plot Apparent Power
plots(outputs['App Supply'], outputs['App Battery'], outputs['App Solar'], outputs['App Diesel'], 
      'Apparent Power', 'Apparent Power [kVA]')    
# Plot Active Power
plots(outputs['Active Supply'], outputs['Active Battery'], outputs['Active Solar'], outputs['Active Diesel'], 
      'Active Power', 'Active Power [kW]')
# Plot Reactive Power
plots(outputs['Reactive Supply'], outputs['Reactive Battery'], outputs['Reactive Solar'], outputs['Reactive Diesel'], 
      'Reactive Power', 'Reactive Power [kVAR]')