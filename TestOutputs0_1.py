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
