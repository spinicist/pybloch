# -*- coding: utf-8 -*-
"""
Created on Thu May 26 09:56:47 2016

@author: Tobias
"""

import numpy as np
from scipy import linalg

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