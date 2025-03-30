import pandas as pd
import ast
import csv

df = pd.read_csv('data/PTB_XL_data/ptbxl_database.csv')
train_data= []
test_data= []
val_data= [] 
line = 0

CD = {
    "LAFB", "LPFB", "IRBBB", "ILBBB", "CLBBB", "CRBBB", "AVB", "IVCD", "WPW", "3AVB", "2AVB"
}

HYP = {
    "LVH", "RVH", "LAO/LAE", "RAO/RAE", "SEHYP"
}

MI = {
    "AMI", "IMI", "LMI", "PMI", "ALMI", "ASMI", "ILMI", "INJAS", 
    "INJAL", "INJIL", "INJIN", "INJLA", "IPMI", "IPMLI"
}

STTC = {
    "ISCAS", "ISCI", "ISC_", "ISCL", "ISCIL", "ISCIN", "ISCAL", "ISCLA", "ISCAN", 
    "ANEUR", "NST_", "NDT", "DIG", "LNGQT", "EL"
}

NORM = {
    "NORM"
}

SUPER_CLASSES = ["NORM", "CD", "HYP", "MI", "STTC"]



df['scp_codes'] = df['scp_codes'].apply(ast.literal_eval)

def max_val(common_with, super_class, data_dict):
    global common
    max_val = 0
    if common_with:
        for sub in common_with:
            if i[sub] > max_val:
                max_val = i[sub]/100
                if max_val > 0:
                    common = True   
    data_dict[super_class] = max_val      


ides = []
for i in df['scp_codes']:
    dict_keys = set(i.keys())
    common_with_norm = dict_keys & NORM
    common_with_cd = dict_keys & CD
    common_with_hyp = dict_keys & HYP
    common_with_mi = dict_keys & MI
    common_with_sttc = dict_keys & STTC
    data_dict = {}



    max_val(common_with_norm, "NORM", data_dict)
    max_val(common_with_cd, "CD", data_dict)
    max_val(common_with_hyp, "HYP", data_dict)
    max_val(common_with_mi, "MI", data_dict)
    max_val(common_with_sttc, "STTC", data_dict)
    # Removed ecg_id temporarily from csv file.
    #data_dict["id"] = df["ecg_id"][line]
    if not common:
        ides.append(df["ecg_id"][line])
    else:
        if df["strat_fold"][line] == 9:
            val_data.append(data_dict)
        elif  df["strat_fold"][line] == 10:
            test_data.append(data_dict)
        else:
            train_data.append(data_dict)

    line += 1
    common = False

with open('data/PTB_XL_data/validation_data.csv', 'w', newline='') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=SUPER_CLASSES)
    writer.writeheader()
    writer.writerows(val_data)

with open('data/PTB_XL_data/test_data.csv', 'w', newline='') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=SUPER_CLASSES)
    writer.writeheader()
    writer.writerows(test_data)

with open('data/PTB_XL_data/train_data.csv', 'w', newline='') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=SUPER_CLASSES)
    writer.writeheader()
    writer.writerows(train_data)