"""
Little PROSPECT interface with some quality-of-life calls.
"""

from src.prospect import prospect_d as PD

n_range = (0.8, 2.5)
cab_range = (0.0, 80.0)

n_default = 1.5
cabn_default = 32
carn_default = 8
cbrownn_default = 0.
cwn_default = 0.016
cmn_default = 0.009
antn_default = 0.0


def get_default_prospect_leaf():
    wls, r, t = PD.run_prospect(
        n=n_default,
        cab=cabn_default,
        car=carn_default,
        cbrown=cbrownn_default,
        cw=cwn_default,
        cm=cmn_default,
        ant=antn_default,
        nr=None, kab=None, kcar=None, kbrown=None, kw=None,
        km=None, kant=None, alpha=40.)
    return wls,r,t


def get_default_prospect_leaf_dry():
    wls, r, t = PD.run_prospect(
        n=n_default,
        cab=cabn_default,
        car=carn_default,
        cbrown=cbrownn_default,
        cw=cwn_default*0.1,
        cm=cmn_default,
        ant=antn_default,
        nr=None, kab=None, kcar=None, kbrown=None, kw=None,
        km=None, kant=None, alpha=40.)
    return wls, r, t
