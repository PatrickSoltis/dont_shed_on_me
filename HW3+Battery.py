import numpy as np
import matplotlib.pyplot as plt
from cvxpy import *
import pandas as pd


#LETS PUSH THIS CHANGE TO GIT WOOOOOOT

#testing testing - Jack
#third time is the charm

## 14 Node Parameters
## baroo

# testing - maya

### Node (aka Bus) Data
'''
Battery: Add 14th node (id=13) with solar (node 9) parent.
Multiple rows in l_P and l_Q to indicate additional points in time.
Populate multiple rows with sample data, or simulate with reasonable variation around central tendency.
Battery consumption is 0 (since discharging in this scenario).
'''

# l_j^P: Active power consumption [MW]
l_P = np.array([[0, 0.2, 0, 0.4, 0.17, 0.23, 1.155, 0, 0.17, 0.843, 0, 0.17, 0.128, 0],
              [0, 0.2, 0, 0.4, 0.17, 0.23, 1.155, 0, 0.17, 0.843, 0, 0.17, 0.128, 0],
              [0, 0.2, 0, 0.4, 0.17, 0.23, 1.155, 0, 0.17, 0.843, 0, 0.17, 0.128, 0],
              [0, 0.2, 0, 0.4, 0.17, 0.23, 1.155, 0, 0.17, 0.843, 0, 0.17, 0.128, 0]])

# l_j^Q: Reactive power consumption [MVAr]
l_Q = np.array([[0, 0.116, 0, 0.29, 0.125, 0.132, 0.66, 0, 0.151, 0.462, 0, 0.08, 0.086, 0],
              [0, 0.116, 0, 0.29, 0.125, 0.132, 0.66, 0, 0.151, 0.462, 0, 0.08, 0.086, 0],
              [0, 0.116, 0, 0.29, 0.125, 0.132, 0.66, 0, 0.151, 0.462, 0, 0.08, 0.086, 0],
              [0, 0.116, 0, 0.29, 0.125, 0.132, 0.66, 0, 0.151, 0.462, 0, 0.08, 0.086, 0]])

# l_j^S: Apparent power consumption [MVA] 
l_S = np.sqrt(l_P**2 + l_Q**2)

# s_j,max: Maximal generating power [MW]
s_max = np.array([5, 0, 0, 3, 0, 0, 0, 0, 0, 3, 0, 0, 0, 3])
# 3 MW power rating on battery

# c_j: Marginal generation cost [USD/MW]
c = np.array([100, 0, 0, 150, 0, 0, 0, 0, 0, 50, 0, 0, 0, 0])
# 0 marginal cost for discharge (all costs in charging, which is outside scope)

# V_min, V_max: Minimum and maximum nodal voltages [V]
v_min = 0.95
v_max = 1.05

'''
Add "juice" as total energy in battery. (MWh)
Must divide juice values by 4 to convert (MW->MWh), assuming each time reading is 15 minute interval.
Begin by assuming battery always has same total energy, but could implement SOC to account for variation.
Could test for optimal size of batter by shifting value of juice and re-running simulation.
'''

juice = 5

'''
r, x, I, and A matrices not yet adjusted to reflect additional node
14th node (id=13) has solar (node 9) parent. Rearrange with 13 in place of 9 and 9 as child?
'''

### Edge (aka Line) Data
# r_ij: Resistance [p.u.]
r = np.array([[0, 0.007547918, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0.0041, 0, 0.007239685, 0, 0.007547918, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0.004343811, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0.003773959, 0, 0, 0.004322245, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0.00434686, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.004343157, 0.01169764],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

# x_ij: Reactance [p.u.]
x = np.array([[0, 0.022173236, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0.0064, 0, 0.007336076, 0, 0.022173236, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0.004401645, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0.011086618, 0, 0, 0.004433667, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0.002430473, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.004402952, 0.004490848],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

# I_max_ij: Maximal line current [p.u.]
I_max = np.array([[0, 3.0441, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 1.4178, 0, 0.9591, 0, 3.0441, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 3.1275, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0.9591, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 3.0441, 3.1275, 0, 0.9591, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 1.37193, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.9591, 1.2927],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

# A_ij: Adjacency matrix; A_ij = 1 if i is parent of j
A = np.array([[0,1,0,0,0,0,0,0,0,0,0,0,0],
                [0,0,1,0,1,0,1,0,0,0,0,0,0],
                [0,0,0,1,0,0,0,0,0,0,0,0,0],
                [0,0,0,0,0,0,0,0,0,0,0,0,0],
                [0,0,0,0,0,1,0,0,0,0,0,0,0],
                [0,0,0,0,0,0,0,0,0,0,0,0,0],
                [0,0,0,0,0,0,0,1,1,0,1,0,0],
                [0,0,0,0,0,0,0,0,0,0,0,0,0],
                [0,0,0,0,0,0,0,0,0,1,0,0,0],
                [0,0,0,0,0,0,0,0,0,0,0,0,0],
                [0,0,0,0,0,0,0,0,0,0,0,1,1],
                [0,0,0,0,0,0,0,0,0,0,0,0,0],
                [0,0,0,0,0,0,0,0,0,0,0,0,0]])

### Set Data
# List of node indices
j_idx = np.arange(14)

# \rho(j): Parent node of node j
rho = np.array([0, 0, 1, 2, 1, 4, 1, 6, 6, 8, 6, 10, 10, 9])

#Adaptation of problem 4

# Assumptions:
#   - Add back all previously disregarded terms and constraints
#   - Relax squared line current equation into inequality
#   - Goal is to minimize generation costs, given by c^T s

# Solve with CVXPY

'''
Must add additional dimension to all variables to account for multiple points in time.
Add constraints for battery: s[13] (summed across all time periods, *0.25 conversion) <= juice
'''

# Define optimization vars
p = Variable(14)
q = Variable(14)
s = Variable(14)
P = Variable(shape=(14,14))
Q = Variable(shape=(14,14))
L = Variable(shape=(14,14))
V = Variable(14)

# Define objective function
objective = Minimize( c.T@s )

# Define constraints
# Apparent Power Limits
constraints = [ s <= s_max ]

# Nodal voltage limits
constraints += [ v_min**2 <= V,
               V <= v_max**2 ]

# Squared line current limits
constraints += [ L <= I_max**2 ]

# Boundary condition for power line flows
constraints += [P[0,0] == 0,
                Q[0,0] == 0]

# Boundary condition for squared line current
constraints += [L[0,0] == 0]

# Fix node 0 voltage to be 1 "per unit" (p.u.)
constraints += [V[0] == 1]


# Loop over each node
for jj in j_idx:
    
    # Parent node, i = \rho(j)
    i = rho[jj]
    
    # Line Power Flows
    constraints += [ p[jj] + P[i,jj] == l_P[jj] + r[i,jj]*L[i,jj] + A[jj,:]@P[jj,:].T,
                   q[jj] + Q[i,jj] == l_Q[jj] + x[i,jj]*L[i,jj] + A[jj,:]@Q[jj,:].T ]

    # Nodal voltage
    constraints += [ V[jj] - V[i] == (r[i,jj]**2 + x[i,jj]**2)*L[i,jj] - 2*(r[i,jj]*P[i,jj].T + x[i,jj]*Q[i,jj].T) ]
    
    # Squared current magnitude on lines
    constraints+= [ quad_over_lin(vstack([P[i,jj],Q[i,jj]]), V[jj]) <= L[i,jj] ]
    
    # Compute apparent power from active & reactive power
    constraints += [ norm(vstack([p[jj],q[jj]])) <= s[jj] ]

# Define problem and solve
prob4 = Problem(objective, constraints)
prob4.solve()

# Output Results
print("------------------- PROBLEM 4 --------------------")
print("--------------------------------------------------")
print(prob4.status)
print("Minimum Generating Cost : %4.2f"%(prob4.value),"USD")
print(" ")
print("Node 0 [Grid]  Gen Power : p_0 = %1.3f"%(p[0].value), "MW | q_0 = %1.3f"%(q[0].value), "MW | s_0 = %1.3f"%(s[0].value),"MW || mu_s0 = %3.0f"%(constraints[0].dual_value[0]), "USD/MW")
print("Node 3 [Gas]   Gen Power : p_3 = %1.3f"%(p[3].value), "MW | q_3 = %1.3f"%(q[3].value), "MW | s_3 = %1.3f"%(s[3].value),"MW || mu_s4 = %3.0f"%(constraints[0].dual_value[3]), "USD/MW")
print("Node 9 [Solar] Gen Power : p_9 = %1.3f"%(p[9].value), "MW | q_9 = %1.3f"%(q[9].value), "MW | s_9 = %1.3f"%(s[9].value),"MW || mu_s9 = %3.0f"%(constraints[0].dual_value[9]), "USD/MW")
print(" ")
print("Total active power   : %1.3f"%(np.sum(l_P)),"MW   consumed | %1.3f"%(np.sum(p.value)),"MW   generated")
print("Total reactive power : %1.3f"%(np.sum(l_Q)),"MVAr consumed | %1.3f"%(np.sum(q.value)),"MVAr generated")
print("Total apparent power : %1.3f"%(np.sum(l_S)),"MVA  consumed | %1.3f"%(np.sum(s.value)),"MVA  generated")
print(" ")
for jj in j_idx:
    print("Node %2.0f"%(jj), "Voltage : %1.3f"%((V[jj].value)**0.5), "p.u.")
print(" ")
for jj in j_idx:
    print("mu_V%2.0f = %2.0f" % (jj, constraints[1].dual_value[jj]))
print(" ")
for jj in j_idx:
    i = rho[jj]
    print("mu_L(%2.0f,%2.0f) = %2.0f"%(i, jj, constraints[3].dual_value[i,jj]))
