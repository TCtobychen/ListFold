import numpy as np 

m = np.load('factor_data.npy', allow_pickle = True)

# Index: 51 is True or False
# We change that into 0 or 1

for i in range(len(m)):
    for j in range(len(m[i])):
        if m[i][j][51] == True:
            m[i][j][51] = 1
        if m[i][j][51] == False:
            m[i][j][51] = 0

np.savez_compressed('features_processed.npz', m)