#!/usr/bin/env python
# -*- coding: utf-8 -*-
# =======================================
# File Name: utils.py
# Purpose  : implementation for generating
#            parameters 
# =======================================

import math
import numpy as np
import csv
import sys
import os

np.random.seed(42) # fixed random seed


def l2_cost(x, y):
    """L2 cost
    """
    m, n = x.shape[0], y.shape[0]
    ind = np.indices((m, n))
    cost = ((x[ind[0]] - y[ind[1]]) ** 2).sum(axis=2)
    return cost


def random_cost(num):
    """random cost
    """
    x = np.random.rand(num * 2).reshape((num, 2))
    y = np.random.rand(num * 2).reshape((num, 2))
    return l2_cost(x, y)


def gen_ellipse_2d(num, x1_center=0, x2_center=0, r_x1=1.3, r_x2=0.9, eps=0.1):
    """generate ellipse 2d datapoint
    """
    r = np.random.uniform(0, 2. * math.pi, num)
    dx1 = np.cos(r) + eps / math.sqrt(2.) * np.random.randn(num)
    dx2 = np.sin(r) + eps / math.sqrt(2.) * np.random.randn(num)
    x1 = r_x1 * dx1 + x1_center
    x2 = r_x2 * dx2 + x2_center
    x = np.concatenate((x1.reshape(num, 1), x2.reshape(num, 1)), axis=1)
    return x


def ellipse_cost(num, x1_center=0, x2_center=0, sr_x1=1.3, sr_x2=0.9, tr_x1=0.9, tr_x2=1.1, eps=0.1):
    """ellipse cost
    """
    x = gen_ellipse_2d(num, x1_center, x2_center, sr_x1, sr_x2, eps)
    y = gen_ellipse_2d(num, x1_center, x2_center, tr_x1, tr_x2, eps)
    return l2_cost(x, y)


def gen_caffarelli_2d(num, x1_center=0, x2_center=0, r=1, shift=2):
    """generate caffarelli 2d datapoint
    """
    l = 0
    while l <= num:
        ox1 = np.random.uniform(-r, r, num)
        ox2 = np.random.uniform(-r, r, num)
        mask = ox1 ** 2 + ox2 ** 2 < r ** 2
        dx1, dx2 = ox1[mask], ox2[mask]
        dx1[dx1 < 0.] -= shift
        dx2[dx2 >= 0.] += shift
        n = dx1.size
        x1 = dx1 + x1_center
        x2 = dx2 + x2_center
        x = np.concatenate((x1.reshape(n, 1), x2.reshape(n, 1)), axis=1)
        if l == 0:
            pos = x
        else:
            pos = np.concatenate((pos, x), axis=0)
        l = len(pos)
    return pos[0:num, ]


def caffarelli_cost(num, x1_center=0, x2_center=0, r=1, shift=2):
    """caffarelli cost
    """
    x = gen_caffarelli_2d(num, x1_center, x2_center, r, 0)
    y = gen_caffarelli_2d(num, x1_center, x2_center, r, shift)
    return l2_cost(x, y)


def DOTmark_weight(num, ImageClass):
    """DOTmark dataset weight for mu and nu
    Schuhmacher, Dominic; Schrieber, Jörn; Gottschlich, Carsten (2016): 
    DOTmark v1.0. figshare. Dataset. 
    https://doi.org/10.6084/m9.figshare.4288466.v1
    """
    path = os.getcwd()
    index = np.random.choice(range(1, 10), 2, replace=None)
    w = []
    for i in index:
        with open(path + '/DOT/' + ImageClass + '/data' + str(int(math.sqrt(num)))
                  + '_100' + str(i) + '.csv') as csvfile:
            csv_reader = csv.reader(csvfile)
            for row in csv_reader:
                w.append(row)
        csvfile.close()
    w = np.array(w, np.float32).reshape((2, num))
    mu = w[0, :] / sum(w[0, :])
    nu = w[1, :] / sum(w[1, :])
    return mu, nu


def gen_DOTmark(start_x1, end_x1, start_x2, end_x2, size):
    """generate DOTmark 2d datapoint
    """
    step_x1 = (end_x1 - start_x1) / size
    x1 = np.linspace(start_x1 + step_x1 / 2.0, end_x1 - step_x1 / 2.0, size)
    step_x2 = (end_x2 - start_x2) / size
    x2 = np.linspace(start_x2 + step_x2 / 2.0, end_x2 - step_x2 / 2.0, size)
    x1p, x2p = np.meshgrid(x1, x2) 
    x = np.concatenate((x1p.reshape((int(size * size), 1)), x2p.reshape((int(size * size), 1))), axis=1)
    return x


def DOTmark_cost(size, start_x1=0, end_x1=1, start_x2=0, end_x2=1):
    """DOTmark cost
    """
    x = gen_DOTmark(start_x1, end_x1, start_x2, end_x2, int(math.sqrt(size)))
    y = gen_DOTmark(start_x1, end_x1, start_x2, end_x2, int(math.sqrt(size)))
    return l2_cost(x, y)


def min_row(A, eps):
    """min row
    """
    return - eps * np.log(np.sum(np.exp(-A / eps), axis=1))

def min_col(A, eps):
    """min col
    """
    return - eps * np.log(np.sum(np.exp(-A / eps), axis=0))



def get_params(n, dataset, imageclass="WhiteNoise"):
    """ get mu, nu, c for Optimal Transport problem from given dataset
    """
    if dataset == 'random':
        mu = np.random.random(n)
        mu = mu / sum(mu)
        nu = np.random.random(n)
        nu = nu / sum(nu)
        c = random_cost(n)
    elif dataset == 'caffarelli':
        mu = np.ones(n) / n
        nu = np.ones(n) / n
        c = caffarelli_cost(n)
    elif dataset == 'ellipse':
        mu = np.ones(n) / n
        nu = np.ones(n) / n
        c = ellipse_cost(n, 0, 0, 1.3, 0.9, 0.9, 1.1, 0.1)
    elif dataset == 'DOTmark':
        if n != 1024 and n != 4096:
            print("Unsupported datasize, exit")
            sys.exit(1)
        mu, nu = DOTmark_weight(n, imageclass)
        c = DOTmark_cost(n)
    return mu, nu, c