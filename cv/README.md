# AlexNet

**论文名：** ImageNet Classification with Deep Convolutional Neural Networks

**链接：** http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf

**简介：**  提出了Alexnet网络结构，并将网络拆分到多GPU上训练，使用dropout，提出权重可视化等。

**file_name:**  AlexNet.pdf



# Faster-Rcnn

**论文名：** Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks

**链接：**  https://arxiv.org/abs/1506.01497

**简介：** 目标检测经典论文。提出rpn网络，利用anchor机制与roi pooling获得候选框，再进行分类。

**file_name:**  faster-rcnn.pdf



# Guided Anchoring

**论文名：** Region Proposal by Guided Anchoring

**链接：** https://arxiv.org/abs/1901.03278

**简介:**  目标检测，基于faster-rcnn的结构，添加了几个卷积层用以生成anchor，令网络可以生成合适的anchor。

**file_name:** GuidAnchor.pdf



# SiameseNetwork

**论文名：** Learning a Similarity Metric Discriminatively, with Application to Face Verification

**链接：**http://yann.lecun.com/exdb/publis/pdf/chopra-05.pdf

**简介：** 孪生神经网络。通过两个共享权重的网络，判断两个输入的相似程度，可以用在人脸识别上。

**file_name:**  SiameseNetwork.pdf



# Unet

**论文名：**U-Net: Convolutional Networks for Biomedical Image Segmentation

**链接：** https://arxiv.org/abs/1505.04597

**简介：** 基于FCN的网络结构，修改了网络的框架，浅层与深层网络的特征进行堆叠，通常用于图像分割，可以进行像素级分类。

**file_name:**  Unet.pdf



# Image Style Transfer

**论文名：**Image Style Transfer Using Convolutional Neural Networks

**链接：** https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Gatys_Image_Style_Transfer_CVPR_2016_paper.pdf

**简介：** 风格迁移。基本上可以理解为把另一张图片的纹理迁移到另一张图片上，通过定义风格相似度的loss进行学习。

**file_name:** Image_Style_Transfer.pdf



# Image Transformer

**论文名：** Image Transformer

**链接：** https://arxiv.org/abs/1802.05751

**简介：** 将nlp领域中的Transformer应用到生成图像序列的任务上。

**file_name:**  ImageTransformer.pdf



# mixup

**论文名：** mixup: Beyond Empirical Risk Minimization

**链接：** https://arxiv.org/abs/1710.09412

**简介：** 相当于提出一个数据增广的方法，将训练数据中的数据（包括label）按一定比例进行加权，然后获得新数据，进行训练，可以有效防止过拟合。

**file_name:**   mixup.pdf



# PCN 

**论文名：** Real-Time Rotation-Invariant Face Detection with Progressive Calibration Networks

**链接：** https://arxiv.org/abs/1804.06039

**简介：** 人脸检测的论文，但实际上思路在其他任务上也可以应用。由于卷积神经网络没有旋转不变性，基本上多个角度的图还是死记硬背，这篇论文提出了一个分3阶段去检测的方法，第一个阶段把图片转到[0, 180]，第二个阶段转到[-45, 45]，第三个阶段直接回归角度， 同时3个阶段都进行坐标回归的多任务训练。通过解决几个相对简单的子问题，得到更好的效果。

**file_name:**  pcn.pdf



# STN

**论文名：** Spatial Transformer Networks

**链接：**  https://arxiv.org/abs/1506.02025

**简介：** 空间变换网络。通过一个网络学习一组空间变换的参数，然后利用这个参数对原始输入的图片or 特征图进行变换，得到映射的特征图，利用双线性插值的特点，使得这个映射过程可以得到梯度，从而进行训练。需要注意，stn并不存在旋转不变性，所以不能做旋转矫正。

**file_name:**   SpatialTransformerNetworks.pdf



# RSTN

**论文名：** Recurrent Spatial Transformer Networks

**链接：** https://arxiv.org/abs/1509.05329

**简介：** 在STN的基础上，加上了RNN，使得网络可以关注不同位置的内容，类似于Attention机制。

**file_name:**   RSTN.pdf



# OctConv

**论文名：**Drop an Octave: Reducing Spatial Redundancy in Convolutional Neural Networks with Octave Convolution

**链接：** https://arxiv.org/abs/1904.05049

**简介：** 提出了新的卷积结构，可以提取高频和低频的图像和特征信息，相当于卷积层面的多尺度。

**file_name:**   oct_conv.pdf



# Harmonic Networks

**论文名：** Harmonic Networks: Deep Translation and Rotation Equivariance

**链接：** https://arxiv.org/abs/1612.04642

**简介：** 研究旋转不变性的论文，利用了复数，圆形谐波之类的东西，没大看懂。

**file_name:** harmonic_conv.pdf





# Xception， Depthwise Separable Convolutions

**论文名：** Xception: Deep Learning with Depthwise Separable Convolutions

**链接：** https://arxiv.org/abs/1610.02357

**简介：** 提出了可分离卷积，并在此基础上提出了新的卷积神经网络结构。

**file_name:**  Xception_Depthwise.pdf



# Visualizing and Understanding Convolutional Networks

**论文名：**Visualizing and Understanding Convolutional Networks

**链接：** https://arxiv.org/abs/1311.2901

**简介：** 关于卷积可视化的论文。提出了一个新的卷积可视化方法，利用反卷积、uppooling等操作，可以可视化出任意一个feature关注的特征。

**file_name:**  Visualizing-and-Understanding-Convolutional-Networks.pdf



# IMAGENET-TRAINED CNNS ARE BIASED TOWARDS TEXTURE

**论文名：** IMAGENET-TRAINED CNNS ARE BIASED TOWARDS TEXTURE; INCREASING SHAPE BIAS IMPROVES ACCURACY AND ROBUSTNESS

**链接:**  https://arxiv.org/abs/1811.12231

**简介：** 研究卷积网络学习性质的一篇论文，论文指出，卷积网络可能通过一些简单的特征（如纹理）就可以cover训练集，这时，一些复杂的特征（如形状）就无法被学到，通过添加纹理的增广，可以使网络逐渐学习形状等特征。

**file_name:** IMAGENET-TRAINED-CNNS-ARE-BIASED-TOWARDS-TEXTURE-INCREASING-SHAPE-BIAS-IMPROVES-ACCURACY-AND-ROBUSTNESS.pdf



# Grad-Cam

**论文名：** Grad-CAM: Why did you say that?

**链接：** https://arxiv.org/abs/1611.07450

**简介：** Class Activation Mapping（类激活图），可以用在分类网络的可视化中，得到分类依据的热力图。利用梯度计算feature_map的加权图，解决了CAM需要修改结构重新训练的弊端。

**file_name:** Grads-Cam.pdf