# AutoSDF: Shape Priors for 3D Completion, Reconstruction and Generation
[[`arXiv`](https://arxiv.org/abs/2203.09516)]
[[`Project Page`](https://yccyenchicheng.github.io/AutoSDF/)]
[[`BibTex`](#citation)]

Code release for the CVPR 2022 paper "AutoSDF: Shape Priors for 3D Completion, Reconstruction and Generation".

https://user-images.githubusercontent.com/27779063/159215086-6889da7c-07c6-4909-b51f-bc04364072cf.mp4



# Installation
Please install [`pytorch`](https://pytorch.org/) and [`pytorch3d`](https://github.com/facebookresearch/pytorch3d). Or you can setup the environment using `conda`:


install packages in an existing conda environment
```
conda activate 3dPrinter
conda env update --file /data/3dPrinter/2_AutoSDF-master/autosdf.yaml
```


However, the environments varies by each machine. We tested the code on `Ubuntu 20.04`, `cuda=11.3`, `python=3.8.11`, `pytorch=1.9.0`, `pytorch3d=0.5.0`.

# Demo
We provide a jupyter notebook for demo. First download the pretrained weights from [this link](https://drive.google.com/drive/folders/1n8W_8CfQ7uZDYNrv487sd0oyhRoNLfGo?usp=sharing), and put them under `saved_ckpt`. 

可以在 linux 上 先 pip install gdown, 然后运行:
```
gdown https://drive.google.com/uc?id=1Zy-L9ADw2h4nWJG4QmaWhALk1Cpe1jw8
gdown https://drive.google.com/uc?id=1clfH8AfX90bIxxFw7sHYK-u0aWuahE_f
gdown https://drive.google.com/uc?id=1IzEXuUiE4nt8axizxZK2w7XBp5qq0RyE
gdown https://drive.google.com/uc?id=1ZH3JMGXcO-C8Gp3iTgzmt4Ig8bpgQmCv
```

Then start the notebook server with
```
jupyter notebook
```
And run:
- `demo_shape_comp.ipynb` for shape completion
- `demo_single_view_recon.ipynb` for single-view reconstruction
- `demo-lang-conditional.ipynb` for language-guided generation

# Preparing the Data
## 1. 下载 `ShapeNetCore.v1` 数据集 from [ShapeNet](https://www.shapenet.org)

* 直接百度云 下载 `ShapeNetCore.v1.zip` 
    ```
    链接：https://pan.baidu.com/s/1WnJIAk4slq99GzE08dELqA 
    提取码：aic6 
    ```
* 之后解压缩这个文件&放到`TARGET_DIR` 下面，通过运行:
  * ```
    #!/bin/bash

    # 目标目录
    TARGET_DIR="/data/3dPrinter/0_Dataset_Ori/4_DISN_Datasets/1_Downloaded/ShepeNet/ShapeNetCore_v1"
    
    # 查找并解压所有 .zip 文件
    find "$TARGET_DIR" -name "*.zip" -exec unzip -o -d "$TARGET_DIR" {} \;
    ```
    <img width="374" alt="Screenshot 2024-07-13 at 1 05 33 AM" src="https://github.com/user-attachments/assets/a1d2f78a-d30f-4ce0-919f-63549b65d354">
    
* 创建 symlink 来 link `ShapeNetCore.v1`  到 `data/ShapeNet`, 通过运行脚本 `scripts/1_link_ShapeNetCore_V1.sh`

    <img width="1251" alt="image" src="https://github.com/user-attachments/assets/c6bf3f3a-f3cb-4c4d-8a41-5434560f73f9">

从 https://github.com/laughtervv/DISN/tree/f8206adb45f8a0714eaefcae8433337cd562b82a/data/filelists 下载 filelists 文件夹 并且放到 TARGET_DIR or /data/3dPrinter/3_AutoSDF-master/data/ShapeNet/ShapeNetCore.v1 下面 (这个文件夹里面的文件，主要是用来指定哪些文件用来 train, 哪些文件用来做 eval) => 这个文件会在 datasets/snet_dataset.py 里面用到


## 2. 提取 `ShapeNetCore.v1` 数据集 的 SDF values

* To extract SDF values, we followed the [preprocessing steps from DISN](https://github.com/laughtervv/DISN/blob/master/preprocessing/create_point_sdf_grid.py), 可以通过 https://drive.google.com/file/d/1cHDickPLKLz3smQNpOGXD2W5mkXcy1nq/view 下载 (Source: https://github.com/Xharlie/DISN)

解压缩后的文件如下：

<img width="1437" alt="image" src="https://github.com/user-attachments/assets/f8396343-ba3f-467b-8bf1-c1a72a38fe8e">

* NB. 有一点要注意的是，从 https://drive.google.com/file/d/1cHDickPLKLz3smQNpOGXD2W5mkXcy1nq/view 下载的是 32x32x32 (因为 32768 = 32x32x32)的，但是 AutoSDF 里面用的是 64x64x64 所以会发生 shape 不匹配的问题!!!!

* 创建 symlink 来 link `ShapeNetCore.v1`  到 `data/ShapeNet`, 通过运行脚本 `scripts/2_link_ShapeNetCore_V1-SDF.sh`


    <img width="1228" alt="image" src="https://github.com/user-attachments/assets/d0c90a0c-07a8-46e7-b7ab-62c393ecdd3a">



## 3. 下载 Pix3D 数据集 from [Pix3D](https://github.com/xingyuansun/pix3d)

The Pix3D dataset can be downloaded here: https://github.com/xingyuansun/pix3d.

我下载到了 /data/3dPrinter/0_Dataset_Ori/4_DISN_Datasets/1_Downloaded/pix3d

<img width="1015" alt="image" src="https://github.com/user-attachments/assets/385a0ae1-60b6-43fc-8d85-c378efc401cf">

* Pix3D 数据主要是用来完成 Exp2: Single View Generation, 这个数据集含有 img/class/xxx.png, mask/class/xxx.png, model/class/xxx/model.obj ...，一共有9类。

    ```
    pix3d
    ├── img (不同的 2d-image)
    │   ├── class1
    │   │   ├── 0001.png
    │   │   ├── 0002.png
    │   │   ├── ...
    │   ├── class2
    │   ├── ...
    │   ├── ...
    │   └── class9
    │       ├── 0001.png
    │       ├── 0002.png
    │       ├── ...
    ├── mask (不同的 2d-image 的mask)
    │   ├── class1
    │   │   ├── 0001.png
    │   │   ├── 0002.png
    │   ├── class2
    │   ├── ...
    │   ├── ...
    │   └── class9
    │       ├── 0001.png
    │       ├── 0002.png
    │       ├── ...
    └── model (不同的 model.obj)
        ├── class1 (父类)
        │   └── xxx (子类)
        │       ├── model.obj
        │       ├── 3d_keypoints.txt
        │       ├── voxel.mat
        │       └── xxx.mtl
        ├── class2
        ├── ...
        ├── ...
        └── class9
    ```

# Training

1. First train the `P-VQ-VAE` on `ShapeNet`:

    把 /data/3dPrinter/3_AutoSDF-master/configs/paths.py 里面的 变量 dataroot = "/path/to/your/data/root" 换成 自己的 数据集所在的位置，例如：`dataroot = "/path/to/your/data/root"`

然后运行脚本：
```
bash ./launchers/1_train_pvqvae_snet.sh
```
* 这个脚本会运行 train.py 文件, 这个文件会从 `from configs.paths import dataroot` 中读取 `dataroot`
  
<img width="1548" alt="image" src="https://github.com/user-attachments/assets/053d6d72-9fcb-4d60-9d2d-36dae296ed24">

2. Then extract the code for each sample of ShapeNet (caching them for training the transformer):
```
./launchers/2_extract_pvqvae_snet.sh
```



3. Train the random-order-transformer to learn the shape prior:
```
./launchers/3_train_rand_tf_snet_code.sh
```

4. To train the image marginal on Pix3D, first extract the code for each training data of Pix3D
```
./launchers/4_extract_pvqvae_pix3d.sh
```

5. Train the image marginal on Pix3D
```
./launchers/train_resnet2vq_pix3d_img.sh
```

# Issues and FAQ

## 1. Regarding `mcubes` functions
We originally use the implementation of the marching cubes from this repo: https://github.com/JustusThies/PyMarchingCubes. However, some of the dependencies seems to be outdated and makes the installation troublesome. Currently the quick workaround is installing `mcubes` from https://github.com/pmneila/PyMCubes:
```
pip install PyMCubes
```
and replace all the lines `import marching_cubes as mcubes` in our code with `import mcubes`. 

我的做法是：下载 skimage 之后， 把 `import marching_cubes as mcubes` 都替换成 from skimage import measure 

之后再 把 `mcubes.marching_cubes` 替换成 measure.marching_cubes

## 2. 找不到 ./preprocess/isosurface/computeDistanceField  文件夹

得到 Bugs:

```
[*] creating tmp/for_sdf/sdf/chair_model/isosurf.sdf
[*] trimesh_load: demo_data/chair_model.obj
[*] export_mesh:  tmp/for_sdf/norm_mesh/chair_model/pc_norm.obj
[*] command: ./preprocess/isosurface/computeDistanceField tmp/for_sdf/norm_mesh/chair_model/pc_norm.obj 256 256 256 -s  -e 1.3 -o 0.dist -m 1 -c
[*] command: mv 0.dist tmp/for_sdf/sdf/chair_model/isosurf.sdf
sh: ./preprocess/isosurface/computeDistanceField: No such file or directory
mv: cannot stat '0.dist': No such file or directory
```

解决办法：

访问 https://github.com/Xharlie/DISN/tree/master/isosurface 然后复制到 .preprocess 目录下即可

# <a name="citation"></a>Citing AutoSDF

If you find this code helpful, please consider citing:

```BibTeX
@inproceedings{autosdf2022,
  title={{AutoSDF}: Shape Priors for 3D Completion, Reconstruction and Generation},
  author={Mittal, Paritosh and Cheng, Yen-Chi and Singh, Maneesh and Tulsiani, Shubham},
  booktitle={CVPR},
  year={2022}
}
```

# Acknowledgement
This code borrowed heavily from [Cycle-GAN](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix), [VQ-GAN](https://github.com/CompVis/taming-transformers). Thanks for the efforts for making their code available!
