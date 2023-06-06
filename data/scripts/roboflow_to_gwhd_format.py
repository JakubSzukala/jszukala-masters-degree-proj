"""
Super basic script that was used to convert roboflow exported annotations to the format consistent with gwhd dataset.
"""

import pandas as pd

path = '<path to roboflow csv>'

df = pd.read_csv(path)
df = df.loc[df.loc[:, 'filename'] == '<filename in roboflow format>', :]
print(df.head())

string = ''
for row in df.iterrows():
    string += str(row[1]['xmin']) + ' ' + str(row[1]['ymin']) + ' ' + str(row[1]['xmax']) + ' ' + str(row[1]['ymax']) + ';'
print(string)
