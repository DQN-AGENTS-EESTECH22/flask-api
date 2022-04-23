import numpy as np
from PIL import Image
from sklearn.cluster import KMeans, AgglomerativeClustering
import json
from flask import Flask, request
import os
from io import BytesIO
from flask_cors import CORS

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
            dic[str(i)] = x.tolist()

        print(dic)
        return json.dumps(dic)

    except Exception as e:
        print('EXCEPTION:', str(e))
        return 'Error processing image', 500


if __name__ == "__main__":
    port = int(os.environ.get('PORT', 9000))
    app.run(host='0.0.0.0', port=port)
