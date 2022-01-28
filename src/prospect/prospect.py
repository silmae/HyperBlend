"""
Little PROSPECT interface with some quality-of-life calls.
"""

from src.prospect import prospect_d as PD

n_range = (0.8, 2.5)
cab_range = (0.0, 80.0)


def get_default_prospect_leaf():
    wls, r, t = PD.run_prospect(
        n=1.5,
        cab=32,
        car=8,
        cbrown=0.,
        cw=0.016,
        cm=0.009,
        ant=0.0,
        nr=None, kab=None, kcar=None, kbrown=None, kw=None,
        km=None, kant=None, alpha=40.)
    return wls,r,t
