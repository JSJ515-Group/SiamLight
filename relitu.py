import os
from PIL import Image
import paddle
import numpy as np
import cv2
import matplotlib.pyplot as plt
from draw_features import Res2Net_vd
import paddle.nn.functional as F
import paddle
import warnings

warnings.filterwarnings('ignore')


def draw_CAM(model, img_path, save_path, transform=None, visual_heatmap=False):
    '''
    绘制 Class Activation Map
    :param model: 加载好权重的Pytorch model
    :param img_path: 测试图片路径
    :param save_path: CAM结果保存路径
    :param transform: 输入图像预处理方法
    :param visual_heatmap: 是否可视化原始heatmap（调用matplotlib）
    :return:
    '''
    # 图像加载&预处理
    img = Image.open(img_path).convert('RGB')
    img = img.resize((224, 224), Image.BILINEAR)  # Image.BILINEAR双线性插值
    if transform:
        img = transform(img)
    # img = img.unsqueeze(0)
    img = np.array(img).astype('float32')
    img = img.transpose((2, 0, 1))
    img = paddle.to_tensor(img)
    img = paddle.unsqueeze(img, axis=0)
    # print(img.shape)
    # 获取模型输出的feature/score

    output, features = model(img)

    print('outputshape:', output.shape)
    print('featureshape:', features.shape)

    # lab = np.argmax(out.numpy())
    # 为了能读取到中间梯度定义的辅助函数
    def extract(g):
        global features_grad
        features_grad = g

    # 预测得分最高的那一类对应的输出score
    pred = np.argmax(output.numpy())
    # print('***********pred:',pred)
    pred_class = output[:, pred]
    # print(pred_class)

    features.register_hook(extract)
    pred_class.backward()  # 计算梯度

    grads = features_grad  # 获取梯度
    # print(grads.shape)
    # pooled_grads = paddle.nn.functional.adaptive_avg_pool2d( x = grads, output_size=[1, 1])
    pooled_grads = grads

    # 此处batch size默认为1，所以去掉了第0维（batch size维）
    pooled_grads = pooled_grads[0]
    # print('pooled_grads:', pooled_grads.shape)
    # print(pooled_grads.shape)
    features = features[0]

    # print(features.shape)
    # 最后一层feature的通道数
    for i in range(2048):
        features[i, ...] *= pooled_grads[i, ...]

    heatmap = features.detach().numpy()

    heatmap = np.mean(heatmap, axis=0)
    # print(heatmap)
    heatmap = np.maximum(heatmap, 0)
    # print('+++++++++',heatmap)
    heatmap /= np.max(heatmap)
    # print('+++++++++',heatmap)
    # 可视化原始热力图
    if visual_heatmap:
        plt.matshow(heatmap)
        plt.show()

    img = cv2.imread(img_path)  # 用cv2加载原始图像
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))  # 将热力图的大小调整为与原始图像相同
    heatmap = np.uint8(255 * heatmap)  # 将热力图转换为RGB格式
    # print(heatmap.shape)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)  # 将热力图应用于原始图像
    superimposed_img = heatmap * 0.4 + img  # 这里的0.4是热力图强度因子
    cv2.imwrite(save_path, superimposed_img)  # 将图像保存到硬盘


model_re2 = Res2Net_vd(layers=50, scales=4, width=26, class_dim=4)
# model_re2 = Res2Net50_vd_26w_4s(class_dim=4)
modelre2_state_dict = paddle.load("Hapi_MyCNN.pdparams")
model_re2.set_state_dict(modelre2_state_dict, use_structured_name=True)
use_gpu = True

paddle.set_device('gpu:0') if use_gpu else paddle.set_device('cpu')

model_re2.eval()

draw_CAM(model_re2, 'data/data106772/img/test/629.jpg', 'test3.jpg', transform=None, visual_heatmap=True)