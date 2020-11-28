# FractalDB

## Summary

The repository contains a fractal category search and FractalDB renderer in Python.
The paper is based on the paper:


Hirokatsu Kataoka, Kazushige Okayasu, Asato Matsumoto, Eisuke Yamagata, Ryosuke Yamada, Nakamasa Inoue, Akio Nakamura and Yutaka Satoh, "Pre-training without Natural Images", Asian Conference on Computer Vision (ACCV), 2020. [[Project]](https://hirokatsukataoka16.github.io/Pretraining-without-Natural-Images/) [[PDF]](https://openaccess.thecvf.com/content/ACCV2020/papers/Kataoka_Pre-training_without_Natural_Images_ACCV_2020_paper.pdf) [[Dataset](https://hirokatsukataoka16.github.io/Pretraining-without-Natural-Images/#dataset)] [[Oral](http://hirokatsukataoka.net/pdf/accv20_kataoka_oral.pdf)] [[Poster](http://hirokatsukataoka.net/pdf/accv20_kataoka_poster.pdf)]

## Citation

If you use this code, please cite the following paper:

```bibtex
@inproceedings{KataokaACCV2020,
  author={Kataoka, Hirokatsu and Okayasu, Kazushige and Matsumoto, Asato and Yamagata, Eisuke and Yamada, Ryosuke and Inoue, Nakamasa and Nakamura, Akio and Satoh, Yutaka},
  title={Pre-training without Natural Images},
  booktitle={Asian Conference on Computer Vision (ACCV)},
  year={2020},
}
```

## Requirements

* Python 3 (worked at 3.7)

## Execution file

We prepared execution file ```exe.sh``` in the top directory. The execution file contains our recommended parameters. Please type the following commands on your environment.

```bash
chmod +x exe.sh
./exe.sh
```

## Fractal Category Search

Run the code ```param_search/ifs_search.py``` to create fractal categories and their representative images. In our work, the basic parameters are ```--rate 0.2 --category 1000 --numof_point 100000```

```bash
python param_search/ifs_search.py --rate=${fillrate} --category=${numof_category} --numof_point=${numof_point}  --save_dir=${save_dir}
```

The folder structure is constructed as follows.

```misc
./
  data/
    csv_rate20_category1000/
      00000.csv
      00001.csv
      ...
    rate20_category1000/
      00000.png
      00001.png
      ...
  param_search/
  ...
```

## FractalDB Construction

Run the code ```fractal_renderer/make_fractaldb.py``` to construct FractalDB.

```bash
python fractal_renderer/make_fractaldb.py
```

The code includes the following parameters.

```misc
--load_root: Category root with CSV file. You can find in "./data".
--save_root: Create the directory of FractalDB.)
--image_size_x: x-coordinate image size 
--image_size_y: y-coordinate image size
--pad_size_x: x-coordinate padding size
--pad_size_y: y-coordinate padding size
--iteration: #dot/#patch in a fractal image
--draw_type: Rendering type. You can select "{point, patch}_{gray, color}"
--weight_csv: Weight parameter. You can find "./fractal_renderer/weights"
--instance: #instance. 10 -> 1000 instances per category, 100 -> 10,000 instances per category')
```

Moreover, we prepared a script of multi-thread processing in the execution file. Please change the comment-out in the part of execution file as follows.
```misc
''' <- Comment-out the single-thread FractalDB creation
# Create FractalDB
python fractal_renderer/make_fractaldb.py \
    --load_root='./data/csv_rate'${fillrate}'_category'${numof_category} --save_root='./data/FractalDB-'${numof_category} \
    --image_size_x=${imagesize} --image_size_y=${imagesize} --iteration=${numof_ite} --draw_type=${howto_draw} \
    --weight_csv='./fractal_renderer/weights/weights_'${weight}'.csv'
'''

# Multi-thread processing
for ((i=0 ; i<40 ; i++))
do
    python fractal_renderer/make_fractaldb.py \
        --load_root='./data/csv_rate'${fillrate}'_category'${numof_category}'_parallel/csv'${i} \
        --save_root='./data/FractalDB-'${numof_category} --image_size_x=${imagesize} --image_size_y=${imagesize} \
        --iteration=${numof_ite} --draw_type=${howto_draw} --weight_csv='./fractal_renderer/weights/weights_'${weight}'.csv' &
done
wait
```
The number (40) means 40 threads in processing. Please change the number and structure in ```data/csv_rate0.2_category1000_parallel```.

The structure of rendered FractalDB is constructed as follows.

```misc
./
  data/
    FractalDB-1000/
      00000/
        00000_00_count_0_flip0.png
        00000_00_count_0_flip1.png
        00000_00_count_0_flip2.png
        00000_00_count_0_flip3.png
        ...
      00001/
        00001_00_count_0_flip0.png
        00001_00_count_0_flip1.png
        00001_00_count_0_flip2.png
        00001_00_count_0_flip3.png
        ...
  ...
```