#!/usr/bin/env python
# -*- coding: utf-8 -*-
# =======================================
# File Name: sinkhorn.py
# Purpose  : implementation for sinkhorn 
#            algorithm
# =======================================

from utils import get_params
import numpy as np
import sys


def sinkhorn(mu, nu, c, iters=1000, eps=1-2):
    """sinkhorn
    """
    m, _ = c.shape
    v = np.ones(m)
    K = np.exp(-c / eps)

    for _ in range(iters):
        u = mu / np.dot(K, v)
        v = nu / np.dot(K.T, u)
    
    pi = np.diag(u).dot(K).dot(np.diag(v))
    print('error_mu = %.5e' % np.linalg.norm(pi.sum(axis = 1) - mu, 1))
    print('error_nu = %.5e' % np.linalg.norm(pi.sum(axis = 0) - nu, 1))
    print('fval     = %.5e' % (c * pi).sum())


if __name__ == '__main__':
    try:
        print("Test...")
        _mu, _nu, _c = get_params(64, 'random')
        sinkhorn(_mu, _nu, _c)
    except KeyboardInterrupt:
        print ("  Ctrl+C pressed...")
        sys.exit(1)