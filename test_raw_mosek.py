#!/usr/bin/env python
# -*- coding: utf-8 -*-
# =======================================
# File Name: test_raw_mosek.py
# Purpose  : test raw mosek solver
# =======================================

from utils import get_params
from raw_mosek import *
import argparse
import mosek
import time
import sys


"""Parser
"""
parser = argparse.ArgumentParser()
parser.add_argument('--n', type=int, default=64)
parser.add_argument('--dataset', type=str, choices=['random', 'caffarelli', 'ellipse', 'DOTmark'], default='random')
parser.add_argument('--imageclass', type=str, default='WhiteNoise')
parser.add_argument('--method', type=str, choices=['primal', 'dual', 'interior'], default='primal')

args = parser.parse_args()


def main():
    """Main routine
    """
    print("\nTesting raw_mosek")
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
        mosek_primal_simplex(mu, nu, c)
    elif args.method == 'dual':
        mosek_dual_simplex(mu, nu, c)
    elif args.method == 'interior':
        mosek_interior_point(mu, nu, c)
    t = time.time() - start
    print('time     = %.5e' % t)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print ("  Ctrl+C pressed...")
        sys.exit(1)