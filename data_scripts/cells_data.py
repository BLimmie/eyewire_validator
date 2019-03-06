import pandas as pd
import json

df1 = pd.read_csv('cells1.csv')
df2 = pd.read_csv('cells2.csv')
df3 = pd.read_csv('cells3.csv')
df = pd.concat((df1,df2,df3))

df = df[df["Status"] == "Completed"]
df = df[~df["Name"].str.startswith("SN-")]
df = df[~df["Name"].str.startswith("DC-")]
df = df[~df["Name"].str.startswith("IO-")]
df = df[~df["Name"].str.startswith("Old")]
df = df[~df["Name"].str.startswith("nMLF")]
df = df[~df["Name"].str.startswith("RC-")]
df = df[~df["Name"].str.startswith("Abd-")]
df = df[~df["Name"].str.startswith("Treach")]
df = df[~df["Name"].str.startswith("Bomni")]
df = df[~df["Name"].str.startswith("aa296")]
df = df[~df["Name"].str.startswith("70083")]
df = df[~df["Name"].str.startswith("70087")]
df = df[~df["Name"].str.startswith("70153")]
df = df[~df["Name"].str.startswith("70070")]
df = df[~df["Name"].str.startswith("70067")]
df = df[~df["Name"].str.startswith("70072")]
df = df[~df["Name"].str.startswith("17022")]
df = df[~df["Name"].str.startswith("17013")]
df = df[~df["Name"].str.startswith("17012")]
df = df[~df["Name"].str.startswith("50004")]
df = df[df["Size"].astype("int") > 0]
df = df[df["Tags"].str.find('"mystic"') == -1]
df = df.reset_index(drop=True)
print(df.head())
print(df.tail())
print(len(df))
df.to_csv("cells.csv")