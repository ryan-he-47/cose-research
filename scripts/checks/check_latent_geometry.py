"""几何测量数值检查：尺度不变性、已知维度、重复点和独立实例计数。"""
import numpy as np
from scripts.review.latent_geometry_metrics import (spectrum, nearest, dimension_from_distances,
    dimension_profile, stratified_bootstrap_accuracy)


def main():
    rng=np.random.default_rng(81)
    x=rng.normal(size=(6000,3))@np.linalg.qr(rng.normal(size=(48,3)))[0].T
    a,b=spectrum(x),spectrum((x+3.2)/.8066658319309046)
    assert abs(a['participation_ratio']-b['participation_ratio'])<1e-10
    assert a['pca']==b['pca']
    d,_=nearest(x,40)
    m=dimension_from_distances(d)
    assert 2.5<m['two_nn_mle']<3.5, m
    assert 2.5<m['mle']['20']['pooled']<3.5, m
    ratios=dimension_from_distances(d*9)
    assert abs(ratios['two_nn_mle']-m['two_nn_mle'])<1e-10
    duplicate=dimension_profile(np.r_[x[:1000],x[:1000]],sizes=(1000,),repeats=1)
    assert duplicate['unique_n']==1000 and duplicate['repeated_vectors']==1000
    acc=stratified_bootstrap_accuracy(np.array([1,1,0,0]),np.array([0,0,1,1]))
    assert acc['macro_recall']==.5 and acc['n']==4
    print({'three_dimensional_two_nn':m['two_nn_mle'],'three_dimensional_mle20':m['mle']['20']['pooled'],
           'scale_and_translation_invariance':'passed','duplicates_and_instance_counts':'passed'})


if __name__=='__main__':
    main()
