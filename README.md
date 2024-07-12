# AutoSDF: Shape Priors for 3D Completion, Reconstruction and Generation
[[`arXiv`](https://arxiv.org/abs/2203.09516)]
[[`Project Page`](https://yccyenchicheng.github.io/AutoSDF/)]
[[`BibTex`](#citation)]

Code release for the CVPR 2022 paper "AutoSDF: Shape Priors for 3D Completion, Reconstruction and Generation".

https://user-images.githubusercontent.com/27779063/159215086-6889da7c-07c6-4909-b51f-bc04364072cf.mp4



# Installation
Please install [`pytorch`](https://pytorch.org/) and [`pytorch3d`](https://github.com/facebookresearch/pytorch3d). Or you can setup the environment using `conda`:

```
conda env create -f autosdf.yaml
conda activate autosdf
```
or
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

* 直接百度云 下载 `ShapeNetCore.v1` 
    ```
    链接：https://pan.baidu.com/s/1WnJIAk4slq99GzE08dELqA 
    提取码：aic6 
    ```
* Put `ShapeNetCore.v1.zip` under `data/ShapeNet`, 之后解压缩这个文件，通过运行:
  * ```
    #!/bin/bash

    # 目标目录
    TARGET_DIR="/data/3dPrinter/0_Dataset_Ori/4_DISN_Datasets/1_Downloaded/ShapeNetCore.v1"
    
    # 查找并解压所有 .zip 文件
    find "$TARGET_DIR" -name "*.zip" -exec unzip -o -d "$TARGET_DIR" {} \;
    ``` 
* We assume the path to the unzipped folder is `data/ShapeNet/ShapeNetCore.v1`.
<img width="374" alt="Screenshot 2024-07-13 at 1 05 33 AM" src="https://github.com/user-attachments/assets/a1d2f78a-d30f-4ce0-919f-63549b65d354">


  
## 2. 提取 `ShapeNetCore.v1` 数据集 的 SDF values
* To extract SDF values, we followed the [preprocessing steps from DISN](https://github.com/laughtervv/DISN/blob/master/preprocessing/create_point_sdf_grid.py).

解压缩后的文件如下：


## 3. 下载 Pix3D 数据集 from [Pix3D](https://github.com/xingyuansun/pix3d)

The Pix3D dataset can be downloaded here: https://github.com/xingyuansun/pix3d.

我下载到了 /data/3dPrinter/0_Dataset_Ori/3_AutoSDF_Datasets/pix3d

<img width="408" alt="Screenshot 2024-07-10 at 3 08 11 PM" src="https://github.com/qingpowuwu/AutoSDF/assets/140480316/50c904cd-bac6-4005-9b74-782560300338">


# Training

1. First train the `P-VQ-VAE` on `ShapeNet`:

把 /data/3dPrinter/3_AutoSDF-master/configs/paths.py 里面的 变量 dataroot = "/path/to/your/data/root" 换成 自己的 数据集所在的位置，例如：`dataroot = "/path/to/your/data/root"`


```
bash ./launchers/1_train_pvqvae_snet.sh
```

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
