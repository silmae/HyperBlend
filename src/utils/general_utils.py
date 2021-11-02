"""
Utility functions for all scripts in the project.

"""

from numpy.polynomial import Polynomial


def chunks(lst, n):
    """Yield successive n-sized chunks from lst.

    Example
    -----
        original_list = [0,1,2,3,4,5,6,7,8,9]
        chunked_list = list(chunks(original_list, 3))
        -> [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]
    """
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def fit_poly(x,y,degree,name):
    fit = Polynomial.fit(x, y, deg=degree, domain=[0, 1])
    coeffs = fit.convert().coef
    print(f"fitting coeffs for {name}: {coeffs}")
    # y = np.array([np.sum(np.array([coeffs[i] * (j ** i) for i in range(len(coeffs))])) for j in x])
    # plt.plot(x, y, color='black')
    return coeffs
