#!/usr/bin/env python
# -*- coding: utf-8 -*-
# =======================================
# File Name: raw_gurobi.py
# Purpose  : implementation for raw gurobi
#            solver
# =======================================

from utils import get_params
from gurobipy import *
import numpy as np
import sys


def gurobi_set_model(mu, nu, c, M):
    """Setting gurobi model
    """
    m, n = c.shape

    ans = M.addVars(m, n, lb=0., ub=GRB.INFINITY)

    M.addConstrs(LinExpr([(1., ans[i, j]) for j in range(n)]) == mu[i] for i in range(m))
    M.addConstrs(LinExpr([(1., ans[i, j]) for i in range(m)]) == nu[j] for j in range(n))

    M.setObjective(LinExpr([(c[i, j], ans[i, j]) for i in range(m) for j in range(n)]))

    return ans


def solve_gurobi(mu, nu, c, mtd=-1):
    """Base routine for calling gurobi solver
    """
    m, n = c.shape

    M = Model("OT")
    M.setParam('OutputFlag', 0)
    M.setParam(GRB.Param.Method, mtd)

    s = gurobi_set_model(mu, nu, c, M)

    M.optimize()

    sx = M.getAttr("x", s)
    ans = np.array([sx[i, j] for i in range(m) for j in range(n)]).reshape(m, n)
    print('error_mu = %.5e' % np.linalg.norm(ans.sum(axis = 1) - mu, 1))
    print('error_nu = %.5e' % np.linalg.norm(ans.sum(axis = 0) - nu, 1))
    print('fvall    = %.5e' % (c * ans).sum())
    return ans


def gurobi_primal_simplex(mu, nu, c):
    """Primal Simplex Method
    """    
    return solve_gurobi(mu, nu, c, mtd=0)


def gurobi_dual_simplex(mu, nu, c):
    """Dual Simplex Method
    """
    return solve_gurobi(mu, nu, c, mtd=1)


def gurobi_interior_point(mu, nu, c):
    """Interior Point Method
    """
    return solve_gurobi(mu, nu, c, mtd=2)


if __name__ == '__main__':
    try:
        print("Test...")
        _mu, _nu, _c = get_params(64, 'random')
        gurobi_primal_simplex(_mu, _nu, _c)
    except KeyboardInterrupt:
        print ("  Ctrl+C pressed...")
        sys.exit(1)