import pandas as pd
import ast

df = pd.read_csv('ptbxl_database.csv')

opt = ["NORM", "AVB", "IRBBB", "CRBBB", "ILBBB", "CLBBB", "AFIB", "STACH"]

# Omvandla strängar i 'scp_codes' till dictionaries
df['scp_codes'] = df['scp_codes'].apply(ast.literal_eval)

# Filtrera rader där minst en nyckel i scp_codes finns i opt-listan
filtered_df = df[df['scp_codes'].apply(lambda d: any(key in opt for key in d.keys()))]

# Skriv ut de filtrerade raderna
print(filtered_df)
