#!/usr/bin/env python
# -*- coding: utf-8 -*-
# =======================================
# File Name: ADMM_primal.py
# Purpose  : implementation for ADMM method
#            for solving primal problem
# =======================================

from utils import get_params
import numpy as np
import sys


def ADMM_primal(mu, nu, c, iters=10000, rho=1024, alpha=1.618):
    """ADMM_primal
    """
    # initialize
    m, n = c.shape
    pi = np.zeros((m, n))
    pi_dag = np.zeros((m, n))
    w = np.zeros((m, n))
    u = np.zeros(m)
    v = np.zeros(n)

    rho_tilde = rho * 32
    while rho_tilde >= rho:
        for _ in range(iters):
            r = ((-w + u.reshape((m, 1)) + v.reshape((1, n)) - c) / rho + 
                mu.reshape((m, 1)) + nu.reshape((1, n)) + pi_dag)
        
            pi = (r - ((r.sum(axis=1) - r.sum() / (m + n + 1)) / (n + 1)).reshape((m, 1))
                - ((r.sum(axis=0) - r.sum() / (m + n + 1)) / (m + 1)).reshape((1, n)))

            pi_dag = np.maximum(pi + w / rho, 0.0)

            u = u + alpha * rho * (mu - pi.sum(axis=1))
            v = v + alpha * rho * (nu - pi.sum(axis=0))
            w = w + alpha * rho * (pi - pi_dag)

            rho_tilde = rho_tilde / 2

        print('error_mu = %.5e' % np.linalg.norm(pi_dag.sum(axis = 1) - mu, 1))
        print('error_nu = %.5e' % np.linalg.norm(pi_dag.sum(axis = 0) - nu, 1))
        print('fvall    = %.5e' % (c * pi_dag).sum())


if __name__ == '__main__':
    try:
        print("Test...")
        _mu, _nu, _c = get_params(64, 'random')
        ADMM_primal(_mu, _nu, _c)
    except KeyboardInterrupt:
        print ("  Ctrl+C pressed...")
        sys.exit(1)

