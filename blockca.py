#!/usr/bin/env python
# -*- coding: utf-8 -*-
# =======================================
# File Name: blockca.py
# Purpose  : implementation for block
#            coordinate ascent method
# =======================================

from utils import *
import numpy as np
import sys


def blockca(mu, nu, c, iters=1000, eps=1e-2):
    """block coordinate ascent
    """
    m, n = c.shape

    f = np.ones((m, 1))
    g = np.ones((n, 1))

    K = np.exp(- c / eps)

    for _ in range(iters):
        f = min_row(c - g.T, eps) + eps * np.log(mu)
        f = f.reshape((m, 1))
        g = min_col(c - f, eps) + eps * np.log(nu)
        g = g.reshape((n, 1))

    u = np.exp(f.squeeze() / eps)
    v = np.exp(g.squeeze() / eps)
    pi = np.diag(u).dot(K).dot(np.diag(v))
    print('error_mu = %.5e' % np.linalg.norm(pi.sum(axis = 1) - mu, 1))
    print('error_nu = %.5e' % np.linalg.norm(pi.sum(axis = 0) - nu, 1))
    print('fval     = %.5e' % (c * pi).sum())


if __name__ == '__main__':
    try:
        print("Test...")
        _mu, _nu, _c = get_params(64, 'random')
        blockca(_mu, _nu, _c)
    except KeyboardInterrupt:
        print ("  Ctrl+C pressed...")
        sys.exit(1)