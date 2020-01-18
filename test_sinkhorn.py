#!/usr/bin/env python
# -*- coding: utf-8 -*-
# =======================================
# File Name: test_sinkhorn.py
# Purpose  : test sinkhorn algorithm for 
#            entropy regularized problem
# =======================================

from utils import get_params
from sinkhorn import sinkhorn
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
parser.add_argument('--iters', type=int, default=1000)
parser.add_argument('--eps', type=float, default=1)

args = parser.parse_args()


def main():
    """Main routine
    """
    print("\nTesting Sinkhorn")
    print("====================")
    print("m = n  : ", args.n)
    print("dataset: ", args.dataset)
    if args.dataset == 'DOTmark':
        print("class  : ", args.imageclass)
    print("====================")

    mu, nu, c = get_params(args.n, args.dataset, args.imageclass)

    start = time.time()
    sinkhorn(mu, nu, c, args.iters, args.eps)
    t = time.time() - start
    print('time     = %.5e' % t)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print ("  Ctrl+C pressed...")
        sys.exit(1)
