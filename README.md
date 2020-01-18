# Optimal_Transport

Large Scale Computational Optimal Transport



## Code

```
/DOT            # DOTmark dataset
/tools          # shell scripts for batch processing

utils.py        # utility functions

raw_mosek.py    # implement raw_mosek(), directly call mosek solver
raw_gurobi.py   # implement raw_gurobi(), directly call gurobi solver
ADMM_primal.py  # implement ADMM_primal()
ADMM_dual.py    # implement ADMM_dual()
sinkhorn.py     # implement sinkhorn()
blockca.py      # implement blockca()

test_*.py       # test wrapers
```



## Doc

```
/img            # images

doc.pdf      	# doc
doc.tex      	# core tex file

content-1.tex   # tex file for section-1
content-2.tex   # tex file for section-2
content-3.tex   # tex file for section-3
content-4.tex   # tex file for section-4
content-5.tex   # tex file for section-5
content-6.tex   # tex file for section-6
content-x.tex   # tex file for section-x

arxiv.sty       # tex style file
```



## Environment

+ Ubuntu 18.04
+ Python 3.7.3
+ Anaconda 4.8.0