#!/bin/bash

# To run,
# timestamp=$(date +"%Y-%m-%d_%H-%M-%S")
# ./exec/scan_RAA.sh 2>&1 | tee ./exec/logs/run_$timestamp.log

# Test avoid problem

avoid_arg="-sf -m A -et gap_avoid"
ann_arg="-a -g 0.99"

# ./py_gl.sh sim_naive_RAA.py $avoid_arg $ann_arg -lr 1e-4 -n scan_avoid_anneal_lr1e-4

# ./py_gl.sh sim_naive_RAA.py $avoid_arg -lr 1e-4 -n scan_avoid_g9999_lr1e-4

# ./py_gl.sh sim_naive_RAA.py $avoid_arg $ann_arg -mu 1000000 -lr 5e-4 -n scan_avoid_anneal_lr5e-4_mu100k

# ./py_gl.sh sim_naive_RAA.py $avoid_arg -mu 1000000 -lr 5e-4 -n scan_avoid_lr5e-4_mu100k

./py_gl.sh sim_naive_RAA.py $avoid_arg -mc 500000 -n scan_avoid_mu50k

./py_gl.sh sim_naive_RAA.py $avoid_arg -mc 1000000 -n scan_avoid_mu100k

./py_gl.sh sim_naive_RAA.py $avoid_arg -mc 2000000 -n scan_avoid_mu200k

./py_gl.sh sim_naive_RAA.py $avoid_arg $ann_arg -mc 500000 -n scan_avoid_anneal_mu50k

./py_gl.sh sim_naive_RAA.py $avoid_arg $ann_arg -mc 1000000 -n scan_avoid_anneal_mu100k

./py_gl.sh sim_naive_RAA.py $avoid_arg $ann_arg -mc 2000000 -n scan_avoid_anneal_mu200k

# ./py_gl.sh sim_naive_RAA.py $avoid_arg -mu 1000000 -lr 1e-4 -n scan_avoid_g9999_lr1e-4_mu100k

# ./py_gl.sh sim_naive_RAA.py $avoid_arg $ann_arg -mc 100000 -mu 1000000 -lr 1e-4 -n scan_avoid_anneal_lr1e-4_mu100k_mc100k

# ./py_gl.sh sim_naive_RAA.py $avoid_arg -mc 100000 -mu 1000000 -lr 1e-4 -n scan_avoid_g9999_lr1e-4_mu100k_mc100k

# ./py_gl.sh sim_naive_RAA.py $avoid_arg $ann_arg -mc 100000 -arc 128 64 32 -mu 1000000 -lr 1e-4 -n scan_avoid_anneal_lr1e-4_mu100k_mc100k_3L

# ./py_gl.sh sim_naive_RAA.py $avoid_arg -mc 100000 -arc 128 64 32 -mu 1000000 -lr 1e-4 -n scan_avoid_g9999_lr1e-4_mu100k_mc100k_3L

