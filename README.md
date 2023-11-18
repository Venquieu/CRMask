
#  CRMask: Mask Compressed Representation for Instance Segmentation
Paper link: [CRMask](https://pan.baidu.com/s/1mjo6txKytsWjLVByw9Tj9A?pwd=3h5o)  code：3h5o

Here is my Master's thesis, which focus on Instance Segmentation. Specifically, the main idea is a compressed representation method of instance masks.

Based on the method, I built(emm... two years ago) a model named CRMask. The model is based on [AdelaiDet](https://github.com/aim-uofa/AdelaiDet), which is an excellent open source toolbox.

## Introduction
Mainstream single-shot instance segmentation frameworks
represent the masks with low-dimensional information. In
this paper, we propose a new representation method that represents
a mask with three compact representation units and
reconstructs the mask with a non-parametric decoder. Our
method can be plugged into any object detector without prior
information. Based on the method, we introduce an efficient
single-shot framework for instance segmentation. Our framework,
termed CRMask, extracts the units from the network
and reconstructs them into the masks with the decoder. The
CRMask achieves 36.2% mask AP with ResNet101-FPNDCN,
using a single-model with single-scale testing on the
COCO dataset. To demonstrate the generality of our method,
we embed it in BlendMask to reconstruct the attention maps.
The improved BlendMask achieves 1.5% mask AP gains than
the original model with faster inference speed, which convincingly
proves the generality of our representation method.

## Requirements
First install Detectron2 following the official guide: [INSTALL.md](https://github.com/facebookresearch/detectron2/blob/master/INSTALL.md).

Please use Detectron2 with commit id [9eb4831](https://github.com/facebookresearch/detectron2/commit/9eb4831f742ae6a13b8edb61d07b619392fb6543) if you have any issues related to Detectron2.

Then build CRMask with:
```
git clone https://github.com/Venquieu/CRMask.git
cd CRMask
python setup.py build develop
```

## training
```
OMP_NUM_THREADS=1 python tools/train_net.py \
    --config-file configs\CRMask\MS_R_50_1x.yaml \
    --num-gpus 4 \
    OUTPUT_DIR output/SVDMask_MS_R_50_1x
```

## demo
You can download model from [here](https://pan.baidu.com/s/1ygphnpNskEa5xqPdOFyuTA?pwd=jhpe).

```
python demo/demo.py \
    --config-file configs\CRMask\MS_R_101_3x.yaml \
    --input input1.jpg input2.jpg \
    --opts output\SVDMask_MS_R_101_3x\model_final.pth
```