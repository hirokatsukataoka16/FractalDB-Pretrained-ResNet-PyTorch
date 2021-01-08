
numof_category=1000
fillrate=0.2
weight=0.4
imagesize=362
numof_point=100000
numof_ite=200000
howto_draw='patch_gray'
arch=resnet50

# Parameter search
python param_search/ifs_search.py --rate=${fillrate} --category=${numof_category} --numof_point=${numof_point} --save_dir='./data'

# Create FractalDB
python fractal_renderer/make_fractaldb.py \
    --load_root='./data/csv_rate'${fillrate}'_category'${numof_category} --save_root='./data/FractalDB-'${numof_category} \
    --image_size_x=${imagesize} --image_size_y=${imagesize} --iteration=${numof_ite} --draw_type=${howto_draw} \
    --weight_csv='./fractal_renderer/weights/weights_'${weight}'.csv'

# FractalDB Pre-training
python pretraining/main.py --path2traindb='data/FractalDB-'${numof_category} --dataset='FractalDB-'${numof_category} --numof_classes=${numof_category} --usenet=${arch}

# Fine-tuning
python finetuning/main.py --path2db='./data/CIFAR10' --path2weight='./data/weight' --dataset='FractalDB-'${numof_category} --ft_dataset='CIFAR10' --numof_pretrained_classes=${numof_category} --usenet=${arch}
