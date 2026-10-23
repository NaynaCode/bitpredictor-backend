"""Hierarchical clustering of features (redundancy analysis) with suggested cluster representatives."""
import numpy as np
import pandas as pd
from scipy import stats
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform

from data import compute_features

NAN_THRESHOLD = 0.30  # features with more missing values (train part) are excluded


def _f(v):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return v if np.isfinite(v) else None


def cluster_features(payload):
    df = pd.DataFrame(payload['data'])
    if 'dates' in df.columns:
        df = df.rename(columns={'dates': 'date'})
    threshold = float(payload.get('threshold', 0.9))
    horizon = int(payload.get('horizon', 5))
    train_share = float(payload.get('train_share', 0.7))

    feats, names = compute_features(df, payload['indicators'], payload.get('transforms', []))
    n_train = int(len(df) * train_share)
    X = feats[names].replace([np.inf, -np.inf], np.nan).iloc[:n_train]
    close = df['close'].astype(float)
    fwd = np.log(close.shift(-horizon) / close).iloc[:n_train]

    excluded, usable = [], []
    for c in names:
        s = X[c]
        if s.notna().sum() == 0:
            excluded.append({'name': c, 'reason': 'No values'})
        elif s.nunique() <= 1:
            excluded.append({'name': c, 'reason': 'Constant'})
        elif s.isna().mean() > NAN_THRESHOLD:
            excluded.append({'name': c, 'reason': f'{s.isna().mean() * 100:.0f}% missing'})
        else:
            usable.append(c)
    if len(usable) < 2:
        raise ValueError('Need at least 2 usable features to cluster.')

    corr = X[usable].corr(method='spearman', min_periods=50)
    ok = corr.notna().any(axis=1)
    for c in corr.index[~ok]:
        excluded.append({'name': c, 'reason': 'Too few overlapping rows'})
    corr = corr.loc[ok, ok]
    cols = list(corr.columns)
    if len(cols) < 2:
        raise ValueError('Not enough overlapping data to cluster.')

    dist = 1 - corr.abs().fillna(0).to_numpy(copy=True)
    np.fill_diagonal(dist, 0)
    dist = np.clip((dist + dist.T) / 2, 0, None)
    link = hierarchy.linkage(squareform(dist, checks=False), method='average')
    labels = hierarchy.fcluster(link, t=1 - threshold, criterion='distance')

    ic = {}
    for c in cols:
        z = pd.concat([X[c], fwd], axis=1).dropna()
        if len(z) > 30 and z.iloc[:, 0].nunique() > 1:
            ic[c] = abs(stats.spearmanr(z.iloc[:, 0], z.iloc[:, 1])[0])
        else:
            ic[c] = 0.0

    absc = corr.abs().to_numpy(copy=True)
    np.fill_diagonal(absc, np.nan)
    kl = pd.DataFrame({
        'cluster': labels.astype(int),
        'abs_ic': pd.Series(ic).fillna(0),
        'nan_share': X[cols].isna().mean(),
        'max_abs_corr': pd.DataFrame(absc, index=cols).max(axis=1),
    }, index=cols)
    kl = kl.sort_values(['cluster', 'abs_ic', 'nan_share'], ascending=[True, False, True])
    kl['rep'] = ~kl.duplicated('cluster')

    clusters = []
    for cid, g in kl.groupby('cluster'):
        clusters.append({
            'id': int(cid),
            'size': int(len(g)),
            'members': [{
                'name': n,
                'abs_ic': _f(r.abs_ic),
                'nan_share': _f(r.nan_share),
                'max_abs_corr': _f(r.max_abs_corr),
                'representative': bool(r.rep),
            } for n, r in g.iterrows()],
        })
    clusters.sort(key=lambda c: (-c['size'], c['id']))

    return {
        'labels': cols,
        'link': link.tolist(),
        'clusters': clusters,
        'excluded': excluded,
        'n_features': len(cols),
        'n_clusters': len(clusters),
        'horizon': horizon,
        'train_rows': int(n_train),
    }