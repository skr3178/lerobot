# %%
import pandas as pd
# %%
df = pd.read_parquet('svla_so101_pickplace/data/chunk-000/file-000.parquet')
# %%
print(df.head())
print(df.info())
# %%
