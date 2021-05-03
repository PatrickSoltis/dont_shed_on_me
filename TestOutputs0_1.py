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
