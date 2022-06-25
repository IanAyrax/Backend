import argparse
import io
from PIL import Image

from werkzeug.utils import secure_filename
import cv2
import json
from flask import Flask, request, jsonify

import classify
import classify2

my_app = Flask(__name__)

@my_app.route('/')
def hello_world():
    return json.dumps([{'text' : 'Hello World!'}])

@my_app.route('/api/predict', methods=["POST"])
def predict():
    if not request.method == "POST" :
        jsonError = {'error': 'Access Denied!'} 
        
        return jsonify(jsonError)	

    if request.files.get("image") :
        image_file = request.files["image"]
        #image_bytes = image_file.read()
        imageFileName = secure_filename(image_file.filename)
        image_file.save('./images/' + imageFileName)
        #img = Image.open(io.BytesIO(image_bytes))
        img = cv2.imread('./images/' + imageFileName)
        #threshold, result = classify.classify(img)
        #returnJson = {'result': result, 'threshold': threshold}
        # returnJson = classify.classify(img)
        returnJson = classify2.classify(img)

        return jsonify(returnJson)
        #return json.dumps([{'result' : result}])

    return "Empty : No Image"

if __name__ == "__main__" :
    my_app.run(host='0.0.0.0', port=4999)