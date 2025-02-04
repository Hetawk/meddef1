import pandas as pd

if not hasattr(pd, 'Int64Index'):
    pd.Int64Index = pd.Index
if not hasattr(pd, 'Float64Index'):
    pd.Float64Index = pd.Index
if not hasattr(pd, 'UInt64Index'):
    pd.UInt64Index = pd.Index

# Patch for missing StringMethods in pandas.core.strings
if not hasattr(pd.core.strings, 'StringMethods'):
    pd.core.strings.StringMethods = type("StringMethods", (), {})
