"""How nonlinear a torus is, from its spectra: used to rank lattices.

A torus is z_a(theta) = sum_m c[a, m] e^{i m.theta}, with z_a = x_a - i p_a (normalized, beam sigmas) and one FFT
matrix c[a] per plane a. Take |c_m|^2 = |c[x, m]|^2 + |c[y, m]|^2. Then
- the action of plane k is I_k = 1/2 sum_m m_k |c_m|^2 (the enclosed area / 2 pi; equal to the J of the torus);
- the linear power is the sum of |c_m|^2 over |m_x|, |m_y| <= 1 (displaced ellipse and coupling included, 2 (I_x + I_y) to 2 %);
- the distortion is D = sum of |c_m|^2 over the nonlinear harmonics, |m_x| > 1 or |m_y| > 1, divided by the linear power.
A linear torus has no nonlinear harmonic, so D = 0 there. Tori that did not converge get NaN.
"""
import numpy as np
from .detect import linear_power


def spectra(result):
    """|c_m|^2 summed over the two planes, and the harmonic numbers m_x, m_y of every entry."""
    power = np.sum(np.abs(result['c'])**2, axis=1)
    Bx, By = power.shape[1]//2, power.shape[2]//2
    mx, my = np.meshgrid(np.arange(-Bx, Bx+1), np.arange(-By, By+1), indexing='ij')
    return power, mx, my


def spectral_actions(result):
    """I[torus, k] = 1/2 sum_m m_k |c_m|^2."""
    power, mx, my = spectra(result)
    I = np.stack([np.sum(mx*power, axis=(1, 2)), np.sum(my*power, axis=(1, 2))], axis=1)/2
    I[~result['converged']] = np.nan
    return I


def distortion(result, lowest=2, highest=np.inf):
    """D[torus] = sum of |c_m|^2 over the nonlinear harmonics of orders |m_x| + |m_y| = lowest..highest, over the linear
    power. distortion(result) is D; distortion(result, 2, 2) only order 2; distortion(result, 3) the orders from 3 up."""
    power, mx, my = spectra(result)
    order = np.abs(mx)+np.abs(my)
    chosen = (order >= lowest) & (order <= highest) & ((np.abs(mx) > 1) | (np.abs(my) > 1))
    D = np.sum(power*chosen, axis=(1, 2))/linear_power(result['c'])
    D[~result['converged']] = np.nan
    return D
