import h5py
from flask import Flask, request, render_template
from keras.models import load_model
from classify import c100_classify
from scipy import misc
from skimage import io
import tensorflow as tf
from keras.models import load_model
import logging
import requests
import efficientnet.keras as efn
import skimage.transform as transform

app = Flask(__name__)
logging.basicConfig(level=logging.DEBUG)

@app.route('/')
def entry_page():
    return render_template('index.html')

@app.route('/predict_object/', methods=['GET', 'POST'])
def render_message():
    saved_model = 'saved_models/cifar_efficientnetb0_model.h5'
    model = load_model(saved_model)
    
    try:
        image = request.files['image']
        print(image)
        app.logger.warn('logged in successfully')
        app.logger.warn('Yeahie Something logged')
        pred = c100_classify(image, model)
        message = "OK, here's what I think this is:"
        data = [{'name':x, 'probability':y} for x,y in zip(pred.iloc[:,0],pred.iloc[:,1])]
    except Exception as e:
        app.logger.warn(str(e))
        app.logger.warn('Yeahie Something logged in exception')
        message = "Something has gone completely wrong, what did you do?!  Try another image."
        data = [{'name':'Error', 'probability':0}]

    return render_template('index.html',
                            message=message,
                           data=data)


if __name__ == '__main__':
    app.run(debug=True)
