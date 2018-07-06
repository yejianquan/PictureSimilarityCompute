import requests
import os
import numpy as np

def getVectors():
    file_name = open('./vectors.txt', 'r')
    lines = file_name.readlines()
    picNames = []
    picFeatures = []
    for idx, line in enumerate(lines):
        if idx % 2 == 0:
            # idx为偶数时存储的是图片名称
            picNames.append(line.replace('\n', ""))
        else:
            # idx为奇数时存储的是特征向量
            picFeatures.append(list(map(float, line.replace('\n', "")[1:-1].split(","))))
    return picNames, picFeatures


def appendToTxt(s1, l1):
    # Hardcode的用以存储特征向量的txt
    file_name = open('./vectors.txt', 'a+')
    # 将图片名写入txt
    file_name.write(s1 + '\n')
    # 将特征向量转换为字符串写入txt
    file_name.write(str(l1) + '\n')
    file_name.close()


def getFeature(pic):
    # Leonardo API URL
    url = 'https://sandbox.api.sap.com/ml/featureextraction/inference_sync'
    # http request的header，记得把<API-KEY>换成你自己的key
    headers = {
        "Accept": "application/json",
        "APIKey": "<API-KEY>"
    }
    # 打开参数指向的图片
    files = {'files': open(pic, 'rb')}
    r = requests.post(url, files=files, headers=headers)
    # 返回图片的特征向量
    return r.json()['predictions'][0]['feature_vector']

def readFeatures():
    # Leonardo API URL
    url = 'https://sandbox.api.sap.com/ml/featureextraction/inference_sync'
    # http request的header，记得把<API-KEY>换成你自己的key
    headers = {
        "Accept": "application/json",
        "APIKey": "<API-KEY>"
    }
    # 所有图片的地址
    pic_path = "./pics/"
    path_list = os.listdir(pic_path)
    for idx, each_path in enumerate(path_list):
        picName = pic_path + each_path
        files = {'files': open(picName, 'rb')}
        r = requests.post(url, files=files, headers=headers)
        # 根据之前得到的JSON结构，获取特征向量
        featureVector = r.json()['predictions'][0]['feature_vector']
        # 根据之前得到的JSON结构，获取图片名
        fileName = r.json()['predictions'][0]['name']
        # 保存至txt中
        appendToTxt(fileName, featureVector)
        print(fileName, featureVector)


# readFeatures()

def findMostSimiliar(fileNames, featureVectors, vector):
    # 将数组都转换为numpy的ndarray，能极大的加速之后的计算
    featureVectors = np.array(featureVectors)
    vector = np.array(vector)
    # 计算目标特征向量和所有特征向量的向量距离
    distances = np.sum(np.square(featureVectors - vector), axis=1)
    # 找出最小距离对应的向量的index
    minIdx = np.where(distances == np.min(distances))[0][0]
    # 返回index对应的图片名
    return fileNames[minIdx]


def matchSimiliar(pic):
    # 提取图片的特征向量
    feature = getFeature(pic)
    # 获取图片库以及鱼片库对应的向量
    fileNames, featureVectors = getVectors()
    # 找到最近似的图片，并返回图片名
    return findMostSimiliar(fileNames, featureVectors, feature)

# 第一次运行需要调用readFeatures来将./pics下的图片全部转换为向量，转换完以后便可注释掉
#readFeatures()
print(matchSimiliar("./testPics/test.jpg"))



