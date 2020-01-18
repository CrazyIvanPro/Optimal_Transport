#!/usr/bin/env python
# -*- coding: utf-8 -*-
# =======================================
# File Name: ADMM_dual.py
# Purpose  : implementation for ADMM method
#            for solving dual problem
# =======================================

from utils import get_params
import numpy as np
import sys


def ADMM_dual(mu, nu, c, iters=10000, rho=1024, alpha=1.618):
    """ADMM_dual
    """
    # initialize
    m, n = c.shape
    u = np.zeros(m)
    v = np.zeros(n)
    ksi = c - u.reshape((m, 1)) - v.reshape((1, n))
    w = np.zeros((m, n))

    rho_tilde = rho * 16
    while rho_tilde >= rho:
        for _ in range(iters):
            u = ((mu + np.sum(w, axis=1)) / rho - np.sum(v) - 
                    np.sum(ksi, axis=1) + np.sum(c, axis=1)) / n
            v = ((nu + np.sum(w, axis=0)) / rho - np.sum(u)
                  - np.sum(ksi, axis=0) + np.sum(c, axis=0)) / m
            ksi = w / rho + c - u.reshape((m, 1)) - v.reshape((1, n))
            ksi = np.maximum(ksi, 0.0)

            w = w + alpha * (c - u.reshape((m, 1)) - v.reshape((1, n)) - ksi)
            
            rho_tilde = rho_tilde / 2

        pi = -w
        print('error_mu = %.5e' % np.linalg.norm(pi.sum(axis = 1) - mu, 1))
        print('error_nu = %.5e' % np.linalg.norm(pi.sum(axis = 0) - nu, 1))
        print('fvall    = %.5e' % (c * pi).sum())


if __name__ == '__main__':
    try:
        print("Test...")
        _mu, _nu, _c = get_params(64, 'random')
        ADMM_dual(_mu, _nu, _c)
    except KeyboardInterrupt:
        print ("  Ctrl+C pressed...")
        sys.exit(1)

