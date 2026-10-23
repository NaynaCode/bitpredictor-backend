"""Data preparation: technical indicators, sequences, scaling, and time-series split.

Mirrors the frontend indicator set so that experiments are reproducible.
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler


def _sma(s, n): return s.rolling(n).mean()
def _ema(s, n): return s.ewm(span=n, adjust=False).mean()


def _rsi(close, period=14):
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(100)


def _macd(close, fast=12, slow=26, signal=9):
    ef = _ema(close, fast)
    es = _ema(close, slow)
    macd = ef - es
    sig = _ema(macd, signal)
    return macd, sig, macd - sig


def _stoch(high, low, close, k=14, d=3):
    ll = low.rolling(k).min()
    hh = high.rolling(k).max()
    pct_k = ((close - ll) / (hh - ll) * 100).fillna(50)
    return pct_k, pct_k.rolling(d).mean()


def _cci(high, low, close, period=20):
    tp = (high + low + close) / 3
    sma = tp.rolling(period).mean()
    md = tp.rolling(period).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
    return (tp - sma) / (0.015 * md.replace(0, np.nan))


def _roc(close, period=10):
    return (close - close.shift(period)) / close.shift(period) * 100


def _bollinger(close, period=20, nstd=2):
    mid = _sma(close, period)
    sd = close.rolling(period).std()
    upper = mid + nstd * sd
    lower = mid - nstd * sd
    width = (upper - lower) / mid.replace(0, np.nan)
    return upper, mid, lower, width


def _wilder(series, period):
    return series.ewm(alpha=1 / period, adjust=False).mean()


def _adx(high, low, close, period=14):
    tr = pd.concat([(high - low), (high - close.shift()).abs(),
                    (low - close.shift()).abs()], axis=1).max(axis=1)
    up = high.diff()
    down = -low.diff()
    plus_dm = up.where((up > down) & (up > 0), 0.0)
    minus_dm = down.where((down > up) & (down > 0), 0.0)
    atr = _wilder(tr, period).replace(0, np.nan)
    plus_di = 100 * _wilder(plus_dm, period) / atr
    minus_di = 100 * _wilder(minus_dm, period) / atr
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return _wilder(dx, period)


def _atr(high, low, close, period=14):
    tr = pd.concat([(high - low), (high - close.shift()).abs(),
                    (low - close.shift()).abs()], axis=1).max(axis=1)
    return _wilder(tr, period)


def _obv(close, volume):
    return (np.sign(close.diff()).fillna(0) * volume).cumsum()


def _mfi(high, low, close, volume, period=14):
    tp = (high + low + close) / 3
    mf = tp * volume
    pos = mf.where(tp > tp.shift(), 0.0).rolling(period).sum()
    neg = mf.where(tp < tp.shift(), 0.0).rolling(period).sum()
    return (100 - 100 / (1 + pos / neg.replace(0, np.nan))).fillna(100)


INDICATOR_DEFS = [
    ('RSI', lambda d: {'RSI': _rsi(d['close'])}),
    ('MACD', lambda d: dict(zip(['MACD', 'MACD Signal', 'MACD Hist'], _macd(d['close'])))),
    ('Stochastic', lambda d: dict(zip(['%K', '%D'], _stoch(d['high'], d['low'], d['close'])))),
    ('CCI', lambda d: {'CCI': _cci(d['high'], d['low'], d['close'])}),
    ('ROC', lambda d: {'ROC': _roc(d['close'])}),
    ('Bollinger', lambda d: dict(zip(['BB Upper', 'BB Middle', 'BB Lower', 'BB Width'],
                                     _bollinger(d['close'])))),
    ('EMA', lambda d: {'EMA12': _ema(d['close'], 12), 'EMA26': _ema(d['close'], 26)}),
    ('SMA', lambda d: {'SMA20': _sma(d['close'], 20), 'SMA50': _sma(d['close'], 50)}),
    ('ADX', lambda d: {'ADX': _adx(d['high'], d['low'], d['close'])}),
    ('ATR', lambda d: {'ATR': _atr(d['high'], d['low'], d['close'])}),
    ('OBV', lambda d: {'OBV': _obv(d['close'], d['volume'])}),
    ('MFI', lambda d: {'MFI': _mfi(d['high'], d['low'], d['close'], d['volume'])}),
]


def compute_indicators(df, keys):
    out = pd.DataFrame(index=df.index)
    names = []
    for key, fn in INDICATOR_DEFS:
        if key not in keys:
            continue
        for name, series in fn(df).items():
            out[name] = series
            names.append(name)
    return out, names


INPUT_DEFS = [
    ('open', lambda d: {'Open': d['open']}),
    ('high', lambda d: {'High': d['high']}),
    ('low', lambda d: {'Low': d['low']}),
    ('close', lambda d: {'Close': d['close']}),
    ('volume', lambda d: {'Volume': d['volume']}),
    ('return1', lambda d: {'Return': d['close'].pct_change()}),
    ('logreturn1', lambda d: {'LogReturn': np.log(d['close'] / d['close'].shift(1))}),
    ('return5', lambda d: {'Return5': d['close'].pct_change(5)}),
    ('return10', lambda d: {'Return10': d['close'].pct_change(10)}),
]


def _compute_transform(df, t):
    base = df[t['base']]
    p = int(t.get('period', 20))
    tf = t['transform']
    name = f"{tf.upper()}({t['base']},{p})"
    if tf == 'sma':
        s = base.rolling(p).mean()
    elif tf == 'ema':
        s = base.ewm(span=p, adjust=False).mean()
    elif tf == 'roc':
        s = (base - base.shift(p)) / base.shift(p) * 100
    elif tf == 'pct':
        s = base.pct_change(p)
    elif tf == 'detrend':
        s = base - base.rolling(p).mean()
    elif tf == 'zscore':
        s = (base - base.rolling(p).mean()) / base.rolling(p).std()
    else:
        s = base
    return name, s


def compute_features(df, keys, transforms):
    out = pd.DataFrame(index=df.index)
    names = []
    for key, fn in INPUT_DEFS + INDICATOR_DEFS:
        if key not in keys:
            continue
        for name, series in fn(df).items():
            out[name] = series
            names.append(name)
    ta_keys = [k for k in keys if isinstance(k, str) and k.startswith('TA_')]
    if ta_keys:
        from talib_features import compute_all_talib
        ta_df, _, _ = compute_all_talib(df)
        for k in ta_keys:
            if k[3:] in ta_df.columns:
                out[k] = ta_df[k[3:]].values
                names.append(k)
    for t in (transforms or []):
        name, series = _compute_transform(df, t)
        out[name] = series
        names.append(name)
    return out, names


def scaler_to_params(scaler, kind):
    if kind == 'minmax':
        return {'type': 'minmax',
                'min': scaler.data_min_.tolist(),
                'max': scaler.data_max_.tolist()}
    return {'type': 'standard',
            'mean': scaler.mean_.tolist(),
            'std': scaler.scale_.tolist()}


def apply_scaler(arr, params):
    arr = np.asarray(arr, dtype=np.float32)
    if params['type'] == 'standard':
        return (arr - np.array(params['mean'])) / np.array(params['std'])
    rng = np.array(params['max']) - np.array(params['min'])
    return (arr - np.array(params['min'])) / np.where(rng == 0, 1, rng)


def compute_target_series(close, target_type, custom_code=None):
    """Derive the target series T[t] from close prices based on the target type."""
    if isinstance(target_type, str) and target_type.startswith('custom_'):
        if not custom_code:
            raise ValueError("Custom target selected but no code was provided.")
        ns = {}
        exec(custom_code, {'pd': pd, 'np': np}, ns)
        fn = ns.get('compute')
        if not callable(fn):
            raise ValueError("Custom target code must define a function named 'compute(close)'.")
        return fn(close)
    if target_type == 'price':
        return close.copy()
    if target_type == 'log_return':
        return np.log(close / close.shift(1))
    if target_type == 'sma10':
        return close.rolling(10).mean()
    if target_type == 'sma20':
        return close.rolling(20).mean()
    if target_type == 'ema10':
        return close.ewm(span=10, adjust=False).mean()
    return close.pct_change()  # 'return' (simple next-day return)


def prepare_data(ohlcv, indicators, config):
    df = pd.DataFrame(ohlcv)
    if 'dates' in df.columns:
        df = df.rename(columns={'dates': 'date'})

    feats, feature_names = compute_features(df, indicators, config.get('transforms', []))
    close = df['close']
    feats['date'] = df['date'].values
    mode = config.get('taskMode', 'classification')
    target_type = config.get('targetType', 'return')
    T = compute_target_series(close, target_type, config.get('customTargetCode'))
    if mode == 'regression':
        feats['label'] = T.shift(-1).astype(float).values
    elif target_type in ('return', 'log_return'):
        feats['label'] = (T.shift(-1) > 0).astype(float).values
    else:
        feats['label'] = (T.shift(-1) > T).astype(float).values
    feats = feats.dropna().reset_index(drop=True)

    dates = feats['date'].values
    X_raw = feats[feature_names].values.astype(np.float32)
    y = feats['label'].values.astype(np.float32)

    seq_len = config['sequenceLength']
    X, y_arr, sample_dates = [], [], []
    for i in range(len(X_raw) - seq_len):
        X.append(X_raw[i:i + seq_len])
        y_arr.append(y[i + seq_len])
        sample_dates.append(dates[i + seq_len])
    X = np.array(X)

    split = int(len(X) * (1 - config['testSize']))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y_arr[:split], y_arr[split:]
    test_dates = sample_dates[split:]

    flat = X_train.reshape(-1, X_train.shape[2])
    if config['scaler'] == 'minmax':
        scaler = MinMaxScaler()
    else:
        scaler = StandardScaler()
    scaler.fit(flat)
    X_train_s = scaler.transform(X_train.reshape(-1, X_train.shape[2])).reshape(X_train.shape).astype(np.float32)
    X_test_s = scaler.transform(X_test.reshape(-1, X_test.shape[2])).reshape(X_test.shape).astype(np.float32)

    return (X_train_s, y_train, X_test_s, y_test, feature_names,
            scaler, scaler_to_params(scaler, config['scaler']), test_dates)