import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
import json
from flask import Flask, request
import os
from io import BytesIO
from flask_cors import CORS

from tensorflow.python.keras import preprocessing
#from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.python.keras.preprocessing.text import Tokenizer
from keras.utils import np_utils
import pandas
from tensorflow.keras.models import load_model
import pickle 

import colorsys

app = Flask(__name__)
CORS(app)


def filter_pixels(pixels: np.ndarray, low=0.0, high=1.0) -> np.ndarray:
    pix_mean = pixels.mean(1)
    mask = (low <= pix_mean) & (pix_mean <= high)
    idx = np.arange(len(pixels))
    return pixels[idx[mask]]


def open_img(img):
    '''MUST BE A JPG/JPEG'''

    try:
        img = np.asarray(img)
        imgg = img.reshape((-1, 3)).astype("float32") / 255  # Transform it into an array and normalize 0-255 -> 0-1
        return filter_pixels(imgg, low=0, high=0.99)


    except Exception as e:
        print(e)
    return img


@app.route("/")
def home():
    return 'hello world'




@app.route("/model", methods=['POST'])
def pallete():
    try:
        n_clusters = 5
        if 'images' in request.files:
            images = request.files['images']
        else:
            images = BytesIO(request.get_data())

        img = Image.open(images)
        processed = open_img(img)
        kmeans = KMeans(n_clusters)
        kmeans.fit_predict(processed)
        centers = kmeans.cluster_centers_

        rgb = centers * 255
        rgb = rgb.astype(np.int64)

        
        dic = {}
        for i, x in enumerate(rgb):
            HEX = '#%02x%02x%02x' % (x[0], x[1], x[2]) # HEX
            dic[str(i)] = HEX

        print(dic)
        return json.dumps(dic)

    except Exception as e:
        print('EXCEPTION:', str(e))
        return 'Error processing image', 500


###### TEXT TO COLOR MODEL

# Load the model
model = load_model('tr_model')

# Load trained Tokenizer
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

def scale(n):
    return int(n * 255) 

@app.route("/text", methods=['GET'])
def from_text():
    name = request.args.get("search").lower() ### REQUEST THAT NAME...
    tokenized = tokenizer.texts_to_sequences([name])
    padded = preprocessing.sequence.pad_sequences(tokenized, maxlen=25)
    one_hot = np_utils.to_categorical(padded, num_classes=28)
    pred = model.predict(np.array(one_hot))[0]
    r, g, b = scale(pred[0]), scale(pred[1]), scale(pred[2])
    rgb = np.array([r,g,b],dtype=np.int32)

    hsv = colorsys.rgb_to_hsv(pred[0],pred[1],pred[2])
    hsv2 = ((hsv[0]+0.5)%1,hsv[1],hsv[2])
    rgb2 = colorsys.hsv_to_rgb(hsv2[0],hsv2[1],hsv2[2])
    r2,g2,b2 = int(rgb2[0]*255),int(rgb2[1]*255),int(rgb2[2]*255)
    
    HEX = '#%02x%02x%02x' % (r, g, b) # HEX
    HEX2 = '#%02x%02x%02x' % (r2, g2, b2) # HEX
    dic = {'0': HEX, '1':HEX2}
    print(dic)
    return json.dumps(dic)
    


if __name__ == "__main__":
    port = int(os.environ.get('PORT', 9000))
    app.run(host='0.0.0.0', port=port)
