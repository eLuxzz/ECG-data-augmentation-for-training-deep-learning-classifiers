import pandas as pd
import ast
import csv

df = pd.read_csv('ptbxl_database.csv')

df['scp_codes'] = df['scp_codes'].apply(ast.literal_eval)

def CreateTrainRecord():
    line = 0
    records = []
    for i in df['scp_codes']:
        if df["strat_fold"][line] != 9 and df["strat_fold"][line] != 10 :
            records.append(df["filename_hr"][line])
        line += 1
    with open('RECORDS_TRAIN.txt', 'w', newline='') as file:
        for l in records:
            writer = file.write(l + "\n")

def CreateValidRecord():
    line = 0
    records = []
    for i in df['scp_codes']:
        if df["strat_fold"][line] == 9:
            records.append(df["filename_hr"][line])
        line += 1
    with open('RECORDS_VALID.txt', 'w', newline='') as file:
        for l in records:
            writer = file.write(l + "\n")


def CreateTestRecord():
    line = 0
    records = []
    for i in df['scp_codes']:
        if df["strat_fold"][line] == 10:
            records.append(df["filename_hr"][line])
        line += 1
    with open('RECORDS_TEST.txt', 'w', newline='') as file:
        for l in records:
            writer = file.write(l + "\n")

CreateTestRecord()
CreateValidRecord()