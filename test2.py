import pandas as pd

vids_df = pd.read_excel("tcg_vids.xlsx", header=0, index_col=None)

def vidFN(member : str, df : pd.DataFrame):
    print(df.iloc[:,0])

vidFN("fake", vids_df)