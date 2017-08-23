# -*- coding: utf-8 -*-
"""
Created on Thu May 26 09:56:47 2016

@author: Tobias
"""

import numpy as np
from scipy import linalg, optimize

# pylint rules on var / arg names and numbers do not suit science
# pylint: disable=C0103,R0913,R0914

def Gab(T1, T2, TR, alpha_d):
    """Calculate the parameters G, a & b for SSFP Ellipse"""
    alpha = np.radians(alpha_d)
    E1f = np.exp(-TR/T1)
    E2f = np.exp(-TR/T2)
    G = np.sin(alpha)*(1 - E1f)/(1 - E1f*np.cos(alpha) - E2f**2*(E1f - np.cos(alpha)))
    a = E2f
    b = E2f*(1 - E1f)*(1 + np.cos(alpha)) / (1 - E1f*np.cos(alpha) - E2f**2*(E1f - np.cos(alpha)))
    return (G, a, b)

def Gab_qmt(F, kf, T1f, T2f, T1r, T2r, f0_Hz, TR, Trf, alpha_d):
    """Calculate SSFP Ellipse parameters with qMT"""
    alpha = np.radians(alpha_d)
    E1f = np.exp(-TR/T1f)
    E2f = np.exp(-TR/T2f)
    if F > 0:
        kr = kf / F
    else:
        kr = 0

    fk = np.exp(-TR * (kf + kr))
    fw = np.exp(-W*Trf)
    E1r = np.exp(-TR/T1r)
    G_gauss = (T2r / np.sqrt(2*np.pi))*np.exp(-(2*np.pi*f0_Hz*T2r)**2 / 2)
    w1 = alpha / Trf # Assume rectangular pulses for now
    W = np.pi * G_gauss * w1**2

    A = 1 + F - fw*E1r*(F+fk)
    B = 1 + fk*(F-fw*E1r*(F+1))
    C = F*(1-E1r)*(1-fk)
    Gp = (np.sin(alpha)*((1-E1f)*B+C))/(A - B*E1f*np.cos(alpha) - E2f**2*(B*E1f-A*np.cos(alpha)))
    ap = E2f
    bp = ((E2f*(A-B*E1f)*(1+np.cos(alpha)))/
          (A - B*E1f*np.cos(alpha) - E2f**2*(B*E1f-A*np.cos(alpha))))
    return (Gp, ap, bp)

def signal(G, a, b, f0_Hz, phi, TR):
    """Convert the SSFP Ellipse parameters into a magnetization"""
    psi = 2 * np.pi * f0_Hz * TR
    theta = psi + phi
    m = G*np.exp(1j*psi/2)*(1 - a*np.exp(1j*theta))/(1 - b*np.cos(theta))
    return m

def hyper_ellipse(x, y):
    """Run the Hyper Ellipse algorithm"""
    D = np.column_stack((x*x, x*y, y*y, x, y, np.ones(x.size)))
    S = np.dot(D.T, D)

    C = np.zeros((6, 6))
    N = x.shape[0]
    xc = np.sum(x) / N
    yc = np.sum(y) / N
    sx = np.sum(np.square(x)) / N
    sy = np.sum(np.square(y)) / N
    xy = np.sum(x * y) / N
    C = [[6*sx, 6*xy, sx+sy, 6*xc, 2*yc, 1],
         [6*xy, 4*(sx+sy), 6*xy, 4*yc, 4*xc, 0],
         [sx + sy, 6*xy, 6*sy, 2*xc, 6*yc, 1],
         [6*xc, 4*yc, 2*xc, 4, 0, 0],
         [2*yc, 4*xc, 6*yc, 0, 4, 0],
         [1, 0, 1, 0, 0, 0]]
    (evals, evecs) = linalg.eig(C, S)
    if np.abs(evals[5]) > np.abs(evals[0]):
        Z = np.real(evecs[:, 5])
    else:
        Z = np.real(evecs[:, 0])
    return Z

def Z_to_AB(Z):
    """Convert the Polynomial representation of an ellipse into the semi-major/minor axes and
       ellipse center"""
    za = Z[0]
    zb = Z[1]/2
    zc = Z[2]
    zd = Z[3]/2
    zf = Z[4]/2
    zg = Z[5]
    dsc = (zb**2-za*zc)
    xc = (zc*zd-zb*zf)/dsc
    yc = (za*zf-zb*zd)/dsc
    numer = 2*(za*zf**2+zc*zd**2+zg*zb**2-2*zb*zd*zf-za*zc*zg)
    A = np.sqrt(numer/(dsc*(np.sqrt((za-zc)**2 + 4*zb**2)-(za+zc))))
    B = np.sqrt(numer/(dsc*(-np.sqrt((za-zc)**2 + 4*zb**2)-(za+zc))))
    if B < A:
        A, B = B, A
    return (A, B, xc + 1j*yc)

def AB_to_Gab(A, B, c, below_ernst=False):
    """Convert semi-major/minor axes to Ellipse parameters"""
    if below_ernst:
        b = (c*A + np.sqrt((c*A)**2-(c**2+B**2)*(A**2-B**2)))/(c**2+B**2)
    else:
        b = (-c*A + np.sqrt((c*A)**2-(c**2+B**2)*(A**2-B**2)))/(c**2+B**2)
    a = B / (b*B + c*np.sqrt(1-b**2))
    G = c*(1 - b**2)/(1 - a*b)
    return (G, a, b)

def direct_fit(cdata, phi, TR):
    """Find Ellipse Parameters by optimization (experimental)"""
    def error_func(x):
        (G, a, b, f0_Hz, phi_0) = x
        sig = signal(G, a, b, f0_Hz, phi+phi_0, TR)
        err = np.abs(sig - cdata)
        return err

    c_mean = np.mean(cdata)
    x_init = (np.abs(c_mean), 0.9, 0.9, np.angle(c_mean), 0)
    x_lower = (0, 0, 0, -1/TR, -np.pi)
    x_upper = (1, 1, 1, 1/TR, np.pi)
    result = optimize.least_squares(error_func, x_init, bounds=((x_lower, x_upper)), verbose=0)
    return result.x

def calc_ellipse_pars(cd, phi, TR, below_ernst=False, method='hyper'):
    """Find the Ellipse parameters that best fit a set of measurements"""
    scale = np.max(np.abs(cd))

    if method == 'hyper':
        x = np.squeeze(np.real(cd)/scale)
        y = np.squeeze(np.imag(cd)/scale)
        Z = hyper_ellipse(x, y)
        (A, B, center) = Z_to_AB(Z)
        c = np.abs(center)
        (G, a, b) = AB_to_Gab(A, B, c, below_ernst)
    else:
        (G, a, b, f0_Hz, phi_0) = direct_fit(cd/scale, phi, TR)
    return (G*scale, a, b)

def calc_mri_pars(a, b, TR, FA):
    """Convert Ellipse Parameters into MRI Parameters"""
    al = np.radians(FA)
    T1 = -TR / np.log((a*(1+np.cos(al)-a*b*np.cos(al))-b)/(a*(1+np.cos(al)-a*b)-b*np.cos(al)))
    T2 = -TR / np.log(a)
    return (T1, T2)
