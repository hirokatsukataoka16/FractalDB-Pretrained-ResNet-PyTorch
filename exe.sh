#!/bin/bash

#$ -l rt_F=1
#$ -l h_rt=72:00:00
#$ -j y
#$ -cwd
#$ -m abe
#$ -N FractalDB

source ~/anaconda3/bin/activate pt14

numof_category=1000
fillrate=0.2
weight=0.4
imagesize=362
numof_point=100000
numof_ite=200000
howto_draw='patch_gray'

# Parameter search
python param_search/ifs_search.py --rate=${fillrate} --category=${numof_category} --numof_point=${numof_point} --save_dir='./data'

# Create FractalDB
python fractal_renderer/make_fractaldb.py \
    --load_root='./data/csv_rate'${fillrate}'_category'${numof_category} --save_root='./data/FractalDB-'${numof_category} \
    --image_size_x=${imagesize} --image_size_y=${imagesize} --iteration=${numof_ite} --draw_type=${howto_draw} \
    --weight_csv='./fractal_renderer/weights/weights_'${weight}'.csv'

''' <- Please remove the comments out
# Multi-thread processing
for ((i=0 ; i<40 ; i++))
do
    python fractal_renderer/make_fractaldb.py \
        --load_root='./data/csv_rate0.2_category1000_parallel/csv'${i} \
        --save_root='./data/FractalDB-'${numof_category} --image_size_x=${imagesize} --image_size_y=${imagesize} \
        --iteration=${numof_ite} --draw_type=${howto_draw} --weight_csv='./fractal_renderer/weights/weights_'${weight}'.csv' &
done
wait
'''