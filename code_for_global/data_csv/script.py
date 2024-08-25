import csv
import pandas as pd
import numpy
import os

cell_dataset = pd.read_csv('train_dxp.csv')
# cell_dataset.loc[(cell_dataset[])]
a = list(cell_dataset['image_id'])
# print(len(set(a)))
b = []
for i in a:
    b.append(i.split('_')[0]+'_20')
print(len(b))
cell = set(b)
print(len(cell))
sample_dataset = pd.read_csv("fold_adni_train_test.csv")
d=sample_dataset.loc[(sample_dataset['label']==1)]
# print(d)
c = list(d['name'])
print(len(set(c)-(set(c)-set(b))))
# print(sample_dataset)

