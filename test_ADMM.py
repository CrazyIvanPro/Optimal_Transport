#!/usr/bin/env python
# -*- coding: utf-8 -*-
# =======================================
# File Name: test_ADMM.py
# Purpose  : test ADMM solver for primal
#            problem and dual problem
# =======================================

from utils import get_params
from ADMM_primal import ADMM_primal
from ADMM_dual import ADMM_dual
import numpy as np
import argparse
import time
import sys

"""Parser
"""
parser = argparse.ArgumentParser()
parser.add_argument('--n', type=int, default=64)
parser.add_argument('--dataset', type=str, choices=['random', 'caffarelli', 'ellipse', 'DOTmark'], default='random')
parser.add_argument('--imageclass', type=str, default='WhiteNoise')
parser.add_argument('--method', type=str, choices=['primal', 'dual'], default='primal')
parser.add_argument('--iters', type=int, default=10000)
parser.add_argument('--alpha', type=float, default=1.618)
parser.add_argument('--rho', type=float, default=1024)

args = parser.parse_args()


def main():
    """Main routine
    """
    print("\nTesting ADMM")
    print("====================")
    print("m = n  : ", args.n)
    print("dataset: ", args.dataset)
    if args.dataset == 'DOTmark':
        print("class  : ", args.imageclass)
    print("method : ", args.method)
    print("====================")

    mu, nu, c = get_params(args.n, args.dataset, args.imageclass)

    start = time.time()
    if args.method == 'primal':
        ADMM_primal(mu, nu, c, args.iters, args.rho, args.alpha)
    elif args.method == 'dual':
        ADMM_dual(mu, nu, c, args.iters, args.rho, args.alpha)
    t = time.time() - start
    print('time     = %.5e' % t)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print ("  Ctrl+C pressed...")
        sys.exit(1)