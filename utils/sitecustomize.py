# sitecustomize.py

print("sitecustomize.py loaded")
import pandas as pd

# For Pandas 2.x, ensure these attributes exist
if not hasattr(pd, 'Int64Index'):
    pd.Int64Index = pd.Index
if not hasattr(pd, 'Float64Index'):
    pd.Float64Index = pd.Index
if not hasattr(pd, 'UInt64Index'):
    pd.UInt64Index = pd.Index
