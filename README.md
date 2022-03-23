# FractalDB

## Summary

The repository contains a Fractal Category Search, FractalDB Construction, Pre-training, and Fine-tuning in Python/PyTorch.

The repository is based on the paper:
Hirokatsu Kataoka, Kazushige Okayasu, Asato Matsumoto, Eisuke Yamagata, Ryosuke Yamada, Nakamasa Inoue, Akio Nakamura and Yutaka Satoh, "Pre-training without Natural Images", International Journal of Computer Vision (IJCV) / ACCV 2020 <font color="red">Best Paper Honorable Mention Award</font> [[Project](https://hirokatsukataoka16.github.io/Pretraining-without-Natural-Images/)] [[PDF (IJCV)](https://link.springer.com/content/pdf/10.1007/s11263-021-01555-8.pdf)] [[PDF (ACCV)](https://openaccess.thecvf.com/content/ACCV2020/papers/Kataoka_Pre-training_without_Natural_Images_ACCV_2020_paper.pdf)] [[Dataset](https://hirokatsukataoka16.github.io/Pretraining-without-Natural-Images/#dataset)] [[Oral](http://hirokatsukataoka.net/pdf/accv20_kataoka_oral.pdf)] [[Poster](http://hirokatsukataoka.net/pdf/accv20_kataoka_poster.pdf)]

## Updates

**Update (Mar 23, 2022)**
* The paper was accepted to International Journal of Computer Vision (IJCV). We updated the scripts and pre-trained models in the extended experiments. [[PDF](https://link.springer.com/content/pdf/10.1007/s11263-021-01555-8.pdf)] [[Pre-trained Models](https://drive.google.com/drive/folders/1tTD-cKKEgBjacCi4ZJ6bRYOv6FsjtGt_?usp=sharing)]


**Update (May 22, 2021)**
* Related project "Can Vision Transformers Learn without Natural Images?" was released. We achieved to train vision transformers (ViT) without natural images. [[Project](https://hirokatsukataoka16.github.io/Vision-Transformers-without-Natural-Images/)] [[PDF](https://arxiv.org/abs/2103.13023)] [[Code](https://github.com/nakashima-kodai/FractalDB-Pretrained-ViT-PyTorch)]


**Update (Jan. 8, 2021)**
* Pre-training & Fine-tuning codes
* Downloadable pre-training models [[Link](https://drive.google.com/drive/folders/1tTD-cKKEgBjacCi4ZJ6bRYOv6FsjtGt_?usp=sharing)]
* Multi-thread preparation with ```param_search/parallel_dir.py```
* Divide execution files into single-thread processing ```exe.sh``` and multi-thread processing ```exe_parallel.sh``` for FractalDB rendering.

## Citation

If you use this code, please cite the following paper:

```bibtex
@inproceedings{KataokaIJCV2022,
  author={Kataoka, Hirokatsu and Okayasu, Kazushige and Matsumoto, Asato and Yamagata, Eisuke and Yamada, Ryosuke and Inoue, Nakamasa and Nakamura, Akio and Satoh, Yutaka},
  title={Pre-training without Natural Images},
  article={International Journal on Computer Vision (IJCV)},
  year={2022},
}

@inproceedings{KataokaACCV2020,
  author={Kataoka, Hirokatsu and Okayasu, Kazushige and Matsumoto, Asato and Yamagata, Eisuke and Yamada, Ryosuke and Inoue, Nakamasa and Nakamura, Akio and Satoh, Yutaka},
  title={Pre-training without Natural Images},
  booktitle={Asian Conference on Computer Vision (ACCV)},
  year={2020},
}
```

## Requirements

* Python 3.x (worked at 3.7)
* Pytorch 1.x (worked at 1.4)
* CUDA (worked at 10.1)
* CuDNN (worked at 7.6)
* Graphic board (worked at single/four NVIDIA V100)

* Fine-tuning datasets
If you would like to fine-tune on an image dataset, you must prepare conventional or self-defined datasets. [[This repository](https://github.com/chatflip/ImageRecognitionDataset)] includes a downloader as an optional way. To use the following execution files ```exe.sh``` and ```exe_parallel.sh```, you should set the downloaded CIFAR-10 dataset in ```./data``` as the following structure.

```misc
./
  data/
    CIFAR10/
      train/
        airplane/
          0001.png
          0002.png
          ...
        ...
      val/
        airplane/
          0001.png
          0002.png
          ...
        ...

# Caution! We changed the dir name from 'test' to 'val'
```

## Execution file

We prepared execution files ```exe.sh``` and ```exe_parallel.sh``` in the top directory. The execution file contains our recommended parameters. Please type the following commands on your environment. You can execute the Fractal Category Search, FractalDB Construction, Pre-training, and Fine-tuning.

```bash
chmod +x exe.sh
./exe.sh
```

For a faster execution, you shuold run the ```exe_parallel.sh``` as follows. You must adjust the thread parameter ```numof_thread=40``` in the script depending on your computational resource.

```bash
chmod +x exe_parallel.sh
./exe_parallel.sh
```


## Fractal Category Search

Run the code ```param_search/ifs_search.py``` to create fractal categories and their representative images. In our work, the basic parameters are ```--rate 0.2 --category 1000 --numof_point 100000```

```bash
python param_search/ifs_search.py --rate=0.2 --category=1000 --numof_point=100000  --save_dir='./data'
```

The structure of directories is constructed as follows.

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

## Pre-training

Run the code ```pretraining/main.py``` to create a FractalDB pre-trained model.

```misc
python pretraining/main.py
```

Please confirm a FractalDB is existing in ```./data``` directory. After the pre-training, a trained model is created like ```FractalDB-1000_resnet50_epoch90.pth``` and ```FractalDB-1000_resnet50_checkpoint.pth.tar```. Moreover, you can resume the training from a checkpoint by assigning ```--resume``` parameter. 

These are the important parameters in pre-training.
```misc
--dataset: model name
--path2traindb: path to FractalDB
--path2weight: path to trained weight
--resume: path to latest checkpoint
--usenet: CNN architecture
--epochs: end epoch
--numof_classes: number of pre-trained class
```

**Pre-trained models**
Our pre-trained models are available in this [[Link](https://drive.google.com/drive/folders/1tTD-cKKEgBjacCi4ZJ6bRYOv6FsjtGt_?usp=sharing)].

We have mainly prepared two different pre-trained models. These pre-trained models are trained on FractalDB in different categories (1k and 10k) and the same number of instances (1k).
```misc
FractalDB-1000_resnet50_epoch90.pth: --dataset=FractalDB-1000 --usenet=resnet50 --epochs=90 --numof_classes=1000
FractalDB-10000_resnet50_epoch90.pth: --dataset=FractalDB-10000 --usenet=resnet50 --epochs=90 --numof_classes=10000
```

If you would like to additionally train from the pre-trained model, you command with the next fine-tuning code as follows.
```misc
# FractalDB-1000_resnet50_epoch90.pth
python finetuning/main.py --path2db='/path/to/your/fine-tuning/data' --dataset='FractalDB-1000' --ft_dataset='YourDataset' --numof_pretrained_classes=1000 --usenet=resnet50

# FractalDB-10000_resnet50_epoch90.pth
python finetuning/main.py --path2db='/path/to/your/fine-tuning/data' --dataset='FractalDB-10000' --ft_dataset='YourDataset' --numof_pretrained_classes=10000 --usenet=resnet50
```

## Fine-tuning

Run the code ```finetuning/main.py``` to additionally train any image datasets. However, in order to use the fine-tuning code, you must prepare a fine-tuning dataset (e.g., CIFAR-10/100, Pascal VOC 2012). Please look at ```Requirements``` for a dataset preparation and download option.

```misc
python finetuning/main.py --path2db='/path/to/your/fine-tuning/data' --ft_dataset='YourDataset'
```

These are the important parameters in fine-tuning.
```misc
--dataset: model name (pre-training dataset)
--ft_dataset: model name (fine-tuning dataset)
--path2db: path to fine-tuning dataset
--path2weight: path to trained weight
--resume: path to latest checkpoint
--useepoch: use epoch in pre-training model
--usenet: CNN architecture
--epochs: end epoch
--numof_pretrained_classes: num of pre-training class
--numof_classes: number of pre-trained class
```

Anyway, you must arrange the directories ```train``` and ```val``` under the fine-tuning dataset (or rewrite the phase in data loader ```DBLoader```). The following dataset structure is also written in ```Requirements```.

```misc
./
  data/
    CIFAR10/
      train/
        airplane/
          0001.png
          0002.png
          ...
        ...
      val/
        airplane/
          0001.png
          0002.png
          ...
        ...
```

## Terms of use
The authors affiliated in National Institute of Advanced Industrial Science and Technology (AIST), Tokyo Denki University (TDU), and Tokyo Institute of Technology (TITech) are not responsible for the reproduction, duplication, copy, sale, trade, resell or exploitation for any commercial purposes, of any portion of the images and any portion of derived the data. In no event will we be also liable for any other damages resulting from this data or any derived data.