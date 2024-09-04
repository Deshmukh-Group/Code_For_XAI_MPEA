import random

import pandas as pd
import numpy as np
import csv
import sys

n1 = float(sys.argv[1]) + 5
n2 = float(sys.argv[2]) + 5
n3 = float(sys.argv[3]) + 5
n4 = float(sys.argv[4]) + 5
#n5 = float(sys.argv[5])

#N_total = n1 + n2 + n3 + n4 + n5
#print(n1,n2,n3,n4,n5)
N_total = 100

f1 = n1/N_total
f2 = n2/N_total
f3 = n3/N_total
f4 = n4/N_total


df = pd.read_csv('data_100.dat', header=None, skiprows=15, sep='\s+')
df.columns = ['id', 'type', 'x', 'y', 'z', 'c1', 'c2', 'c3']


natoms = df.shape[0]
v1 = round(f1 * natoms)
v2 = round(f2 * natoms)
v3 = round(f3 * natoms)
v4 = round(f4 * natoms)
v5 = natoms - (v1 + v2 + v3 + v4)

df1 = df.sample(n=v1)
df = df.drop(df1.index)

df2 = df.sample(n=v2)
df = df.drop(df2.index)

df3 = df.sample(n=v3)
df = df.drop(df3.index)

df4 = df.sample(n=v4)
df5 = df.drop(df4.index)

df2.type = 2
df3.type = 3
df4.type = 4
df5.type = 5

df_final = pd.concat([df1, df2, df3, df4, df5])

df_final.to_csv('temp100.dat', header=None, index=None, sep=' ')

df_title = pd.read_csv('Header100.txt', header=None, skip_blank_lines=False)
df_coor = pd.read_csv('temp100.dat', header=None)
df_new = pd.concat([df_title, df_coor])
df_new = df_new.replace({np.nan: None})
df_new.to_csv('data100_use.dat', header=None, index=None, na_rep=" ")
