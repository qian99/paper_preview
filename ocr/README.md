# Attention Ocr

**论文名：** Attention-based Extraction of Structured Information from Street View Imagery

**链接：** https://arxiv.org/abs/1704.03549

**简介：**  end-to-end的OCR识别。cnn提取特征，将位置编码接在特征后，使用attention+lstm解码，得到最终结果。

**file_name:**  attention_ocr.pdf



# Crnn

**论文名：** An End-to-End Trainable Neural Network for Image-based Sequence Recognition and Its Application to Scene Text Recognition

**链接：** https://arxiv.org/abs/1507.05717

**简介：**  单行文字识别。使用cnn提取特征，后面接双向lstm + ctc 求loss，可以识别序列文本，有点是长度没有限制。

**file_name:**  Crnn.pdf



# CTC Loss

**论文名： ** Connectionist Temporal Classification: Labelling Unsegmented Sequence Data with Recurrent Neural Networks

**链接： **  https://www.cs.toronto.edu/~graves/icml_2006.pdf

**简介：**  主要用于对不定长序列做解码，不需要对序列中特定的位置进行标注。论文使用了前向后向的动态规划算法，计算当前输出与label的loss。可以用于音频或图像的解码。

**file_name:**  CTCLoss.pdf



# CRAFT

**论文名： ** Character Region Awareness for Text Detection

**链接：** https://arxiv.org/abs/1904.01941

**简介：**  基于图像分割回归的文本检测模型，网络结构类似UNet，回归文字中心的score与文字之间的中心的score，最后使用图像处理的方法将文本提取出来，并且提出了弱监督学习的训练框架，利用了bounding box，解决标注的困难。

**file_name:**  Craft.pdf



# 

**论文名：**

**链接：**

**简介：**  

**file_name:** 



# 

**论文名：**

**链接：**

**简介：**  

**file_name:** 



# 

**论文名：**

**链接：**

**简介：**  

**file_name:** 



# 

**论文名：**

**链接：**

**简介：**  

**file_name:** 



# 

**论文名：**

**链接：**

**简介：**  

**file_name:** 



# 

**论文名：**

**链接：**

**简介：**  

**file_name:** 



# 

**论文名：**

**链接：**

**简介：**  

**file_name:** 



# 

**论文名：**

**链接：**

**简介：**  

**file_name:** 



# 

**论文名：**

**链接：**

**简介：**  

**file_name:** 