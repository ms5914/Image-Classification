import os
import pickle
import pandas as pd
import numpy as np
from flask import Flask, request, render_template
import matplotlib.pyplot as plt
import cv2
import seaborn as sns

app = Flask(__name__)


def unpickle(file):
    with open(file, 'rb') as fo:
        myDict = pickle.load(fo, encoding='latin1')
    return myDict

def resize_test_image(test_img):
    app.logger.warn('in resize')
    img = cv2.imdecode(np.frombuffer(test_img.read(), np.uint8), cv2.IMREAD_UNCHANGED | cv2.IMREAD_COLOR)
    app.logger.warn('after imread')
    img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    app.logger.warn('after color')
    resized_img = cv2.resize(img_RGB, (224, 224))
    app.logger.warn('after resize')
    resized_img = resized_img / 255.
    #plt.imshow(resized_img)
    return resized_img

# function to get the dataframe for top 5 predictions
def predict_test_image(test_img, model):
    app.logger.warn('in predict ')
    resized_img = resize_test_image(test_img)
    app.logger.warn('after resize ')
    prediction = model.predict(np.array([resized_img]))
    app.logger.warn('after prediction ')

    return prediction


def sort_prediction_test_image(test_img, model):
    app.logger.warn('In sort')
    prediction = predict_test_image(test_img, model)
    app.logger.warn('after predict in sort')
    index = np.arange(0, 100)

    for i in range(100):
        for j in range(100):
            if prediction[0][index[i]] > prediction[0][index[j]]:
                temp = index[i]
                index[i] = index[j]
                index[j] = temp

    return index, prediction


def df_top5_prediction_test_image(test_img, model):
    app.logger.warn('In top5_predict')
    sorted_index, prediction = sort_prediction_test_image(test_img, model)
    app.logger.warn('after sort')
    #prediction = predict_test_image(test_img,model)
    app.logger.warn('after predict')
    metaData = unpickle('/Users/mahakshah/Desktop/cifar100_CNN-master/data/cifar-100-python/meta')
    app.logger.warn('after metadata')
    category = pd.DataFrame(metaData['coarse_label_names'], columns=['SuperClass'])
    # storing fine labels along with its number code in a dataframe
    subCategory = pd.DataFrame(metaData['fine_label_names'], columns=['SubClass'])

    subCategory_name = []
    prediction_score = []

    k = sorted_index[:6]

    for i in range(len(k)):
        subCategory_name.append(subCategory.iloc[k[i]][0])
        prediction_score.append(round(prediction[0][k[i]], 2))

    df = pd.DataFrame(list(zip(subCategory_name, prediction_score)), columns=['Label', 'Probability'])
    return df

def c100_classify(image, model):
    app.logger.warn('In classify')
    return df_top5_prediction_test_image(image, model)


