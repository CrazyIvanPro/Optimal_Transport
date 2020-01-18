#!/usr/bin/env python
# -*- coding: utf-8 -*-
# =======================================
# File Name: raw_mosek.py
# Purpose  : implementation for raw mosek 
#             solver
# =======================================

from utils import get_params
import numpy as np
import mosek
import sys


def mosek_set_model(mu, nu, c, task):
    """Setting mosek model
    """
    task.putobjsense(mosek.objsense.minimize)
    m, n = c.shape

    task.appendvars(m * n)
    task.appendcons(m + n)

    task.putvarboundlist(range(m * n), [mosek.boundkey.lo] * (m * n),
                         [0.0] * (m * n), [0.0] * (m * n))

    for i in range(m):
        task.putarow(i, range(i * n, (i + 1) * n), [1.0] * n)
    task.putconboundlist(range(0, m), [mosek.boundkey.fx] * m, mu, mu)

    for i in range(n):
        task.putarow(i + m, range(i, i + m * n, n), [1.0] * m)
    task.putconboundlist(range(m, m + n), [mosek.boundkey.fx] * n, nu, nu)

    task.putclist(range(m * n), c.reshape(m * n))



def solve_mosek(mu, nu, c, mtd=None, sol=None, log=None):
    """Base routine for calling mosek solver
    """
    m, n = c.shape

    with mosek.Env() as env:
        env.set_Stream(mosek.streamtype.log, log)

        with env.Task() as task:
            task.set_Stream(mosek.streamtype.log, log)
            task.putintparam(mosek.iparam.optimizer, mtd)

            mosek_set_model(mu, nu, c, task)

            task.optimize()

            xx = [0.0] * (m * n)
            task.getxx(sol, xx)

            ans = np.array(xx).reshape(m, n)
            print('error_mu = %.5e' % np.linalg.norm(ans.sum(axis = 1) - mu, 1))
            print('error_nu = %.5e' % np.linalg.norm(ans.sum(axis = 0) - nu, 1))
            print('fvall    = %.5e' % (c * ans).sum())
    return ans


def mosek_primal_simplex(mu, nu, c):
    """Primal Simplex Method
    """
    return solve_mosek(mu, nu, c, 
            mtd = mosek.optimizertype.primal_simplex, sol = mosek.soltype.bas)


def mosek_dual_simplex(mu, nu, c):
    """Dual Simplex Method
    """
    return solve_mosek(mu, nu, c, 
            mtd = mosek.optimizertype.dual_simplex, sol = mosek.soltype.bas)


def mosek_interior_point(mu, nu, c):
    """Interior Point Method
    """
    return solve_mosek(mu, nu, c, 
            mtd = mosek.optimizertype.intpnt, sol = mosek.soltype.itr)


if __name__ == '__main__':
    try:
        print("Test...")
        _mu, _nu, _c = get_params(64, 'random')
        mosek_primal_simplex(_mu, _nu, _c)
    except KeyboardInterrupt:
        print ("  Ctrl+C pressed...")
        sys.exit(1)