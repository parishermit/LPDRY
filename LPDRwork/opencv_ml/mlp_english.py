import cv2 as cv
import numpy as np
import os
# 导入MLPClassifiar
# 导入多层感知机用于分类模型 MLPClassifiar
from sklearn.neural_network import MLPClassifier
# 导入完成模型持久化的joblib
import joblib

# 初始配置
# 训练集、测试集位置
TRAIN_DIR = 'data/enu_train'
TEST_DIR = 'data/enu_test'
# 图片宽、高
IMAGE_WIDTH = 20
IMAGE_HEIGHT = 20
# 输出类别数、类别-数值 字典
CLASSIFICATION_COUNT = 34
LABEL_DICT = {
	'0':0, '1':1, '2':2, '3':3, '4':4, '5':5, '6':6, '7':7, '8':8, '9':9,
	'A':10, 'B':11, 'C':12, 'D':13, 'E':14, 'F':15, 'G':16, 'H':17,	'J':18, 'K':19,
	'L':20, 'M':21, 'N':22, 'P':23, 'Q':24, 'R':25, 'S':26, 'T':27, 'U':28, 'V':29,
	'W':30, 'X':31, 'Y':32, 'Z':33
}

# 模型持久化的位置
MLP_ENU_MODEL_PATH = 'model/mlp/mlp_enu.m'

# 1. 加载数据集
# 参数：数据集所在目录位置： dir_path
# 返回值：样本的特征矩阵、标签向量，features,labels
def load_data(dir_path):
    data = []
    labels = []
    # 获取数据集目录下的所有的子目录并逐一遍历
    for item in os.listdir(dir_path):
        # 获取每一个具体样本类型的os的路径形式
        item_path = os.path.join(dir_path, item)
        # item_path
        # 判断只有目录，才进入进行下一级目录的遍历
        if os.path.isdir(item_path):
            # 每一个样本目录,遍历其下的每一个训练集样本文件-图片
            for subitem in os.listdir(item_path):
                subitem_path = os.path.join(item_path, subitem)
                gray_image = cv.imread(subitem_path, cv.IMREAD_GRAYSCALE)
                resized_image = cv.resize(gray_image, (IMAGE_WIDTH, IMAGE_HEIGHT))
                data.append(resized_image.ravel())
                labels.append(LABEL_DICT[item])
    # 分别赋值 样本数据的特征、样本数据标签
    features = np.array(data)
    labels = np.array(labels)
    # 返回特征矩阵、标签向量
    return features, labels

# 2.预处理 标准化
# 参数：特征矩阵，data
# 返回值：执行标准化后的 data
def normalize_data(data):
    return (data - data.mean()) / data.max()

# 3. 训练 + 保存
def train():
    # 加载训练数据
    print("装载训练数据...")
    train_data, train_labels = load_data(TRAIN_DIR)
    # 预处理
    normalized_data = normalize_data(train_data)
    print("装载%d条数据，每条数据%d个特征" % (normalized_data.shape))

    print("训练中...")
    #模型创建
    model = MLPClassifier(hidden_layer_sizes=(48, 24), solver='lbfgs', alpha=1e-5, random_state=42)

    #模型训练
    model.fit(normalized_data, train_labels)

    str = 'model/mlp'
    if not os.path.exists(str):
        os.makedirs(str)

    # 模型的保存
    print("训练完成，保存模型...")
    joblib.dump(model, MLP_ENU_MODEL_PATH)

# 4. 测试 + 评估
def test():
    print("装载测试数据...")
    test_data, test_labels = load_data(TEST_DIR)
    normalized_data = normalize_data(test_data)
    print("装载%d条数据，每条数据%d个特征" % (normalized_data.shape))

    print("装载模型...")
    model = joblib.load(MLP_ENU_MODEL_PATH)
    print("模型装载完毕，开始测试...")
    predicts = model.predict(normalized_data)

    errors = np.count_nonzero(predicts - test_labels)

    print("测试完毕，预测正确：%d 条，预测错误:%d 条， 正确率：%f" %
          (len(predicts) - errors, errors, (len(predicts) - errors) / len(predicts)))

    # print(errors)

train()
test()