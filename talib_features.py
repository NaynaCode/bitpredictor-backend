"""Compute ALL TA-Lib technical indicators for an OHLCV DataFrame."""
import numpy as np
import pandas as pd

OHLCV = ['open', 'high', 'low', 'close', 'volume']

# Groups / indicators to skip (empty = everything is computed)
EXCLUDED_GROUPS = []
EXCLUDED_INDICATORS = []


def compute_all_talib(df):
    """Returns (DataFrame of indicator columns, {column: group}, {indicator: skip reason})."""
    import talib
    from talib import abstract

    inputs = {c: df[c].to_numpy(dtype='float64') for c in OHLCV if c in df.columns}
    results, groups, skipped = {}, {}, {}

    for group, funcs in talib.get_function_groups().items():
        if group in EXCLUDED_GROUPS:
            continue
        for name in funcs:
            if name in EXCLUDED_INDICATORS:
                continue
            try:
                func = abstract.Function(name)
                needed = []
                for v in func.input_names.values():
                    needed.extend(v if isinstance(v, (list, tuple)) else [v])
                missing = [k for k in needed if k not in inputs]
                if missing:
                    skipped[name] = f'missing inputs: {missing}'
                    continue
                out = func(inputs)
                outputs = func.output_names
                if len(outputs) == 1:
                    out = [out]
                for o, arr in zip(outputs, out):
                    col = name if len(outputs) == 1 else f'{name}_{o}'
                    results[col] = np.asarray(arr, dtype='float64')
                    groups[col] = group
            except Exception as e:  # noqa: BLE001
                skipped[name] = f'error: {e}'

    ind = pd.DataFrame(results, index=df.index).replace([np.inf, -np.inf], np.nan)
    ind = ind.loc[:, ind.notna().any()]
    return ind, {c: groups[c] for c in ind.columns}, skipped