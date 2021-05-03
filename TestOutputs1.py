# %% Debugging

#1 Provides a numbered index for all columns in D_df
#Prerequisites: Parameters A definitions
i = -1
for item in D_df.columns:
    i += 1
    print(i, item)

#2 Compare energy available in system to demand
#Prerequisites: Parameters A&B definitions
print('D:',np.sum(D))
print('s_S:',np.sum(s_S))
print('j_start:',np.sum(j_start))
print('f_start:',np.sum(f_start))
print('energy available:',np.sum([np.sum(s_S),np.sum(j_start),np.sum(f_start)]))

#3 Infeasibility of 3-dimensional variables
P = Variable((lent_t, 8,8)) #active power flow
Q = Variable((lent_t, 8,8)) #reactive power flow
L = Variable((lent_t, 8,8)) #squared magnitude of complex current
#"ValueError: Expressions of dimension greater than 2 are not supported."
#Existing solution: Squeezing line variables into 2 dimensions, since all nodes 1-7 have parent 0.

# %% Variable and Parameter Checks

#Load fraction served
for t in range(len_t):
    print("Time %3.0f"%(t))
    for jj in j_idx:
        print("Node %2.0f fraction served: %1.3f"%(jj, F_P[t,jj].value))

#Load delivered        
for t in range(len_t):
    print("Time %3.0f"%(t))
    for jj in j_idx:
        print("Node %2.0f: l_P: %1.3f\t l_Q:%1.3f"%(jj, l_P[t,jj].value, l_Q[t,jj].value))
        
# Demand
for t in range(len_t):
    print("Time %3.0f"%(t))
    for jj in j_idx:
        print("Node %2.0f demand: %1.3f"%(jj, D_P[t,jj]))
        
# Battery - charge, discharge, and charge state
for t in range(len_t):
    print("Time %3.0f Battery"%(t))
    for jj in j_idx:
        print("Node %2.0f j: %1.3f\tb_gen: %1.3f\tb_eat: %1.3f"%(jj, j[t,jj].value, b_gen[t,jj].value, b_eat[t,jj].value))
        
# Generation - solar, diesel, and diesel fuel stock
for t in range(len_t):
    print("Time %3.0f Generation"%(t))
    for jj in j_idx:
        print("Node %2.0f s_S: %1.3f\td_S: %1.3f\tf: %1.3f"%(jj, s_S[t,jj], d_S[t,jj].value, f[t,jj].value))
        
# Solar resource
for t in range(len_t):
    print("Time %3.0f Power"%(t))
    for jj in j_idx:
        print("Node %2.0f s: %1.3f\ts_max: %1.3f"%(jj, s[t,jj].value, s[t,jj].value))
        
# Real and reactive power generated
for t in range(len_t):
    print("Time %3.0f Power intermediates"%(t))
    for jj in j_idx:
        print("Node %2.0f p: %1.3f\tq: %1.3f"%(jj, p[t,jj].value, q[t,jj].value))
        
# Bus voltages
for t in range(len_t):
    print("Time %3.0f Voltage"%(t))
    for jj in j_idx:
        print("Node %2.0f V: %1.3f"%(jj, V[t,jj].value))
       
# Lines - power and current flows
for t in range(len_t):
    print("Time %3.0f Lines: 0 to node:"%(t))
    for jj in j_idx:
        print("%2.0f P: %1.3f\tQ: %1.3f\tL: %1.3f"%(jj, P[t,jj].value, Q[t,jj].value, L[t,jj].value))
# %%
