"""冻结表示的谱、近邻维度和聚类统计；所有距离均在输入的实际坐标系计算。"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from threadpoolctl import threadpool_limits


def wilson_interval(successes, total, z=1.959963984540054):
    """二项比例的95% Wilson区间；用于固定原型的独立加噪试验。"""
    p=successes/total
    denominator=1+z*z/total
    center=(p+z*z/(2*total))/denominator
    half=z*np.sqrt(p*(1-p)/total+z*z/(4*total*total))/denominator
    return [float(max(0,center-half)),float(min(1,center+half))]


def quantiles(values):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if not len(values):
        return dict(n=0)
    return dict(n=len(values), mean=float(values.mean()),
                **{f'p{p}':float(np.percentile(values,p)) for p in (5,25,50,75,95,99)})


def spectrum(points):
    x = np.asarray(points, np.float64)
    mean = x.mean(0)
    covariance = np.cov(x, rowvar=False)
    eig = np.maximum(np.linalg.eigvalsh(covariance)[::-1], 0)
    p = eig/eig.sum()
    positive = p > 0
    cumulative = np.cumsum(p)
    return dict(n=len(x), mean=mean.tolist(), covariance=covariance.tolist(), eigenvalues=eig.tolist(),
                participation_ratio=float(1/(p@p)),
                covariance_entropy_rank=float(np.exp(-(p[positive]*np.log(p[positive])).sum())),
                pca={str(q):int(np.searchsorted(cumulative,q)+1) for q in (.9,.95,.99)},
                variance_per_coordinate=float(eig.sum()/x.shape[1]),
                centered_norm=quantiles(np.linalg.norm(x-mean,axis=1)),
                origin_norm=quantiles(np.linalg.norm(x,axis=1)),
                snr1_noise_times=(np.sqrt(eig)/(1+np.sqrt(eig))).tolist())


def nearest(points, k=41):
    """double 欧氏距离，排除自身；重复点由调用方显式处理。"""
    x = np.ascontiguousarray(points, np.float64)
    tree = cKDTree(x)
    distances, indices = tree.query(x, k=k+1, workers=4)
    return distances[:,1:], indices[:,1:]


def dimension_from_distances(distances, ks=(10,20,40)):
    d = np.asarray(distances,np.float64)
    valid = np.isfinite(d[:,1]) & (d[:,0] > 0) & (d[:,1] > d[:,0])
    log_ratio = np.log(d[valid,1]/d[valid,0])
    result = dict(two_nn_mle=float((len(log_ratio)-1)/log_ratio.sum()) if len(log_ratio)>1 else None,
                  two_nn_valid=int(valid.sum()), two_nn_tied_or_zero=int((~valid).sum()), mle={})
    for k in ks:
        selected = np.all(np.isfinite(d[:,:k]),axis=1) & (d[:,0]>0)
        values = d[selected,:k]
        denominator = np.log(values[:,-1,None]/values[:,:-1]).sum(1)
        ok = denominator > 0
        local = (k-2)/denominator[ok]
        result['mle'][str(k)] = dict(local=quantiles(local),
            pooled=float((k-2)/denominator[ok].mean()) if ok.any() else None,
            radius=quantiles(values[:,-1]), valid=int(ok.sum()))
    return result


def dimension_profile(points, *, seed=20261103, sizes=(4000,8000,16000), repeats=2):
    x, counts = np.unique(np.asarray(points,np.float64),axis=0,return_counts=True)
    rng = np.random.default_rng(seed)
    result = dict(original_n=len(points), unique_n=len(x), repeated_vectors=int(len(points)-len(x)),
                  largest_exact_multiplicity=int(counts.max()),
                  method='TwoNN Pareto MLE; bias-corrected local LB (k-2)/sum(log(rk/rj)); pooled inverse-log mean',
                  runs=[])
    for n in sorted(set(min(size,len(x)) for size in sizes)):
        if n < 50:
            continue
        for repeat in range(repeats):
            idx = rng.choice(len(x),n,replace=False)
            distances,_ = nearest(x[idx],40)
            result['runs'].append(dict(n=n,repeat=repeat,**dimension_from_distances(distances)))
    return result


def clusters(points, *, seed=20261103, limit=4000):
    rng = np.random.default_rng(seed)
    n = min(limit,len(points))
    x = np.asarray(points,np.float64)[rng.choice(len(points),n,replace=False)]
    cov = np.cov(x,rowvar=False)
    eig, vectors = np.linalg.eigh(cov)
    null = rng.normal(size=x.shape) @ (vectors*np.sqrt(np.maximum(eig,0))).T + x.mean(0)
    result = dict(n=n,ks={},null='single_Gaussian_with_matched_sample_mean_and_covariance')
    with threadpool_limits(limits=4):
        for k in (5,10,20,25):
            labels = KMeans(k,random_state=seed,n_init=5).fit_predict(x)
            null_labels = KMeans(k,random_state=seed,n_init=5).fit_predict(null)
            result['ks'][str(k)] = dict(real=float(silhouette_score(x,labels)),
                                      matched_gaussian=float(silhouette_score(null,null_labels)))
    return result


def stratified_bootstrap_accuracy(correct, labels, *, seed=20261103, repeats=500):
    """每行须是独立字符实例；各类等权。返回宏召回及按类分层实例 bootstrap。"""
    rng = np.random.default_rng(seed)
    classes = np.unique(labels)
    samples = [np.asarray(correct,np.float64)[labels==c] for c in classes]
    per_class = [float(x.mean()) for x in samples]
    distribution = np.zeros(repeats)
    for values in samples:
        distribution += values[rng.integers(len(values),size=(repeats,len(values)))].mean(1)/len(classes)
    return dict(macro_recall=float(np.mean(per_class)), micro_accuracy=float(np.mean(correct)),
                ci95=np.percentile(distribution,[2.5,97.5]).tolist(),
                min_class_n=min(map(len,samples)), max_class_n=max(map(len,samples)),
                classes=len(classes),n=len(correct), per_class=dict(zip(map(str,classes),per_class)),
                interval='within-class instance bootstrap; one row per independently sampled glyph placement/warp')


def grouped_macro_accuracy(correct,labels,units,*,seed=20261103,repeats=300):
    """逐笔指标按整字实例重采样，保留同一字内笔画的相关性。"""
    correct=np.asarray(correct,np.float64)
    classes,label_index=np.unique(labels,return_inverse=True)
    unique,unit_index=np.unique(units,return_inverse=True)
    denominator=np.bincount(label_index,minlength=len(classes))
    numerator=np.bincount(label_index,weights=correct,minlength=len(classes))
    rng=np.random.default_rng(seed);values=[]
    for _ in range(repeats):
        weights=np.bincount(rng.integers(len(unique),size=len(unique)),minlength=len(unique))[unit_index]
        counts=np.bincount(label_index,weights=weights,minlength=len(classes))
        successes=np.bincount(label_index,weights=weights*correct,minlength=len(classes))
        values.append(float((successes/counts).mean()))
    return dict(macro_recall=float((numerator/denominator).mean()),micro_accuracy=float(correct.mean()),
                ci95=np.percentile(values,[2.5,97.5]).tolist(),n=len(correct),independent_glyphs=len(unique),
                classes=len(classes),min_class_n=int(denominator.min()),max_class_n=int(denominator.max()),
                per_class_counts=dict(zip(map(str,classes),map(int,denominator))),
                interval='glyph-instance cluster bootstrap of per-stroke macro recall')
