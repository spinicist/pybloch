# -*- coding: utf-8 -*-
"""
Created on Thu May 26 09:56:47 2016

@author: Tobias
"""

import numpy as np
from scipy import linalg

def Gab(T1, T2, TR, alpha_d):
    alpha = np.radians(alpha_d)
    E1f = np.exp(-TR/T1f)
    E2f = np.exp(-TR/T2f)
    G = np.sin(alpha)*(1 - E1f)/(1 - E1f*np.cos(alpha) - E2f**2*(E1f - np.cos(alpha)))
    a = E2f
    b = E2f*(1 - E1f)*(1 + np.cos(alpha)) / (1 - E1f*np.cos(alpha) - E2f**2*(E1f - np.cos(alpha)))
    return (G, a, b)

def Gab_qmt(F, kf, T1f, T2f, T1r, T2r, f0_Hz, TR, Trf, alpha_d):
    alpha = np.radians(alpha_d)
    E1f = np.exp(-TR/T1f)
    E2f = np.exp(-TR/T2f)
    if F > 0:
        kr = kf / F
    else:
        kr = 0
    G_gauss = (T2r / np.sqrt(2*np.pi))*np.exp(-(2*np.pi*f0_Hz*T2r)**2 / 2)
    w1 = alpha / Trf # Assume rectangular pulses for now
#     W = np.pi * G0 * w1**2
    W = np.pi * G_gauss * w1**2

    fk = np.exp(-TR * (kf + kr))
    fw = np.exp(-W*Trf)
    E1r = np.exp(-TR/T1r)
    E2r = np.exp(-TR/T2r)
    A = 1 + F - fw*E1r*(F+fk)
    B = 1 + fk*(F-fw*E1r*(F+1))
    C = F*(1-E1r)*(1-fk)
    Gp = (np.sin(alpha)*((1-E1f)*B+C))/(A - B*E1f*np.cos(alpha) - E2f**2*(B*E1f-A*np.cos(alpha)))
    ap = E2f
    bp = (E2f*(A-B*E1f)*(1+np.cos(alpha)))/(A - B*E1f*np.cos(alpha) - E2f**2*(B*E1f-A*np.cos(alpha)))
    return (Gp, ap, bp)

def signal(G, a, b, f0_Hz, phi, TR):
    psi = 2 * np.pi * f0_Hz * TR
    theta = psi + phi
    m = G*np.exp(1j*psi/2)*(1 - a*np.exp(1j*theta))/(1 - b*np.cos(theta))
    return m

def calc_ellipse_pars(cd):
    scale = np.max(np.abs(cd))
    x = np.squeeze(np.real(cd)/scale)
    y = np.squeeze(np.imag(cd)/scale)
    D = np.column_stack((x*x, x*y, y*y, x, y, np.ones(x.size)))
    S = np.dot(D.T,D)
    C = np.zeros((6,6))
    
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
    (evals, evecs) = linalg.eig(C,S)
    if (np.abs(evals[5]) > np.abs(evals[0])):
        Z = np.real(evecs[:,5])
    else:
        Z = np.real(evecs[:,0])
    za = Z[0]
    zb = Z[1]/2
    zc = Z[2]
    zd = Z[3]/2
    zf = Z[4]/2
    zg = Z[5]
    
    dsc=(zb**2-za*zc)
    xc = (zc*zd-zb*zf)/dsc
    yc = (za*zf-zb*zd)/dsc
    th = np.arctan2(yc,xc)
    A = np.sqrt((2*(za*zf**2+zc*zd**2+zg*zb**2-2*zb*zd*zf-za*zc*zg))/(dsc*(np.sqrt((za-zc)**2 + 4*zb**2)-(za+zc))))
    B = np.sqrt((2*(za*zf**2+zc*zd**2+zg*zb**2-2*zb*zd*zf-za*zc*zg))/(dsc*(-np.sqrt((za-zc)**2 + 4*zb**2)-(za+zc))))
    print('A ', A, ' B ', B, ' A > B ', A > B)
    if (A > B):
        T=A
        A=B
        B=T
    c = np.sqrt(xc**2+yc**2)
    b = (-2*c*A + np.sqrt((2*c*A)**2-4*(c**2+B**2)*(A**2-B**2)))/(2*(c**2+B**2))
    a =  B / (b*B + c*np.sqrt(1-b**2))
    M = scale*c*(1-b**2)/(1-a*b)
    print(' M ', M, ' a ', a, ' b ', b, ' c ', c)
    return (xc, yc, th, A, B, M, a, b)
    
def calc_mri_pars(a,b,TR,FA):
    al = np.radians(FA)
    T1 = -TR / np.log((a*(1+np.cos(al)-a*b*np.cos(al))-b)/(a*(1+np.cos(al)-a*b)-b*np.cos(al)))
    T2 = -TR / np.log(a)
    return (T1, T2)

def calc_es(data, TR, FA):
    (xc, yc, th, A, B, M, a, b) = calc_ellipse_pars(data)
    (T1, T2) = calc_mri_pars(a, b, TR, FA)
    return (M, T1, T2)