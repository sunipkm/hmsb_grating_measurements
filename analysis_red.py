# %%
from __future__ import annotations
from functools import partial
from typing import SupportsFloat as Numeric, Tuple
import numpy as np
from scipy.optimize import minimize_scalar, minimize, curve_fit
import matplotlib.pyplot as plt
# %%
data = np.loadtxt('data/meas_1.csv', delimiter=',', dtype=float).T
# %%
proc = np.zeros((2, data.shape[-1]))
proc[0] = data[0] + data[1] / 60
proc[1] = data[2] - data[3] / 60
# %%
proc = np.nanmean(proc, axis=0)
dproc = np.full_like(proc, 10/60)
# %%


def func(p: Numeric, a0: Numeric, a1: Numeric | np.ndarray, a2: Numeric | np.ndarray, lam: Numeric = 633e-6, order: int = 1) -> Numeric | np.ndarray:
    """Modified grating equation for m-consecutive orders.
    
    Random coordinate to grating coordinate transformation:
    \\alpha' = \\alpha - \\alpha_0, where angles with prime are in normal coordinates.
    The grating equation is given by:
    \\sin(\\alpha) + \\sin(\\beta) = n \\lambda / p. For two orders n and n + m, we have:
    \\sin(\\beta_{n+m}) - \\sin(\\beta_n) = \\frac{m \\lambda}{p},
    which simplifies to:
    \\frac{m \\lambda}{p} = \\cos(\\frac{\\beta_{n+m}' + \\beta_n'}{2} + \\alpha_0) \\sin(\\frac{\\beta_{n+m}' - \\beta_n'}{2}), in normal coordinates.
    Args:
        p (Numeric): Grating pitch (mm)
        a0 (Numeric): Initial angle (deg)
        a1 (Numeric | np.ndarray): Angle of order n (deg)
        a2 (Numeric | np.ndarray): Angle of order n + m (deg)
        lam (Numeric, optional): Wavelength of light. Defaults to 633e-6 (mm).
        order (int, optional): Order difference (m). Defaults to 1.

    Returns:
        Numeric | np.ndarray: Calculated value of the modified grating equation.
    """
    a0 = np.deg2rad(a0)
    a1 = np.deg2rad(a1)
    a2 = np.deg2rad(a2)
    return (np.cos((a1 + a2) / 2 + a0)*np.sin((a1 - a2) / 2) * 2 * p / lam / order) - 1

# %%
def costfn(x: Tuple[Numeric, Numeric], angles, lam: Numeric = 633e-6, order: int = 1) -> Numeric:
    """Cost function for curve fitting.

    Args:
        p (Numeric): Grating pitch (mm)
        a0 (Numeric): Initial angle (deg)
        a1 (np.ndarray): Angle of order n (deg)
        a2 (np.ndarray): Angle of order n + m (deg)
        lam (Numeric, optional): Wavelength of light. Defaults to 633e-6 (mm).
        order (int, optional): Order difference (m). Defaults to 1.

    Returns:
        Numeric | np.ndarray: Calculated value of the cost function.
    """
    p, a0 = x
    a1 = angles[0:-1]
    a2 = angles[1:]
    ret = func(p, a0, a1, a2, lam, order)
    return np.sum(ret**2)
# %%
ret = minimize(costfn, x0 = (1/98, 90), args=(proc,), bounds=((1/1000, 1), (-180, 180)))
# %%
print(f"Grating pitch: {ret.x[0]:.4f} mm, Grating density: {1/ret.x[0]:.2f} lines/mm")
plt.plot(proc[:-1], func(ret.x[0], ret.x[1], proc[:-1], proc[1:], order=1), 'o', label='Data')
# %%
ret = minimize(partial(costfn, order=2), x0 = (1/98, 90), args=(proc[::2],), bounds=((1/1000, 1), (-180, 180)))
# %%
print(f"Grating pitch: {ret.x[0]:.4f} mm, Grating density: {1/ret.x[0]:.2f} lines/mm")
plt.plot(proc[:-1], func(ret.x[0], ret.x[1], proc[:-1], proc[1:], order=1), 'o', label='Data')
# %%
