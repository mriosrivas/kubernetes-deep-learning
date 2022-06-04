#!/usr/bin/env python
# coding: utf-8
#import time

import numpy as np
import requests
from PIL import Image

import grpc
import tensorflow as tf

from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc

from flask import Flask
from flask import request
#from flask import jsonify
import json



def get_from_url(url):
    r = requests.get(url)

    with open("/tmp/image.jpg", 'wb') as f:
        f.write(r.content)


def preprocess(X):
    return np.float32((X/127.5) - 1)


def get_url(url):
    get_from_url(url)

    with Image.open('/tmp/image.jpg') as img:
        img = img.resize((299, 299), Image.NEAREST)

    X = preprocess(np.array([np.array(img)]))
    
    return X

app = Flask('predict-clothes')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    url = data['url']
    classes = ['dress', 'hat', 'longsleeve', 'outwear', 'pants', 'shirt', 'shoes', 'shorts', 'skirt', 't-shirt']
    print(url)

    #X = get_url('https://bit.ly/3N3qWuF')
    X = get_url(url)

    X_proto = tf.make_tensor_proto(X, shape=X.shape)

    host = 'localhost:8500'
    channel = grpc.insecure_channel(host)

    stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)

    pb_request = predict_pb2.PredictRequest()

    pb_request.model_spec.name = 'clothing-model'
    pb_request.model_spec.signature_name = 'serving_default'

    pb_request.inputs['input_2'].CopyFrom(X_proto)

    pb_response = stub.Predict(pb_request, timeout=20.0)

    prediction = pb_response.outputs['dense_1'].float_val

    result = dict(zip(classes, prediction))

    #return jsonify(prediction)
    return json.dumps(result)


if __name__ == "__main__":
    app.run(debug=True, host='localhost', port=9696)



