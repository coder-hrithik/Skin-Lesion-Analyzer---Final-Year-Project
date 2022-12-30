from flask import Flask, jsonify, request, render_template
from flask_cors import CORS, cross_origin

from keras.models import load_model
import json
import numpy as np
import tensorflow as tf
from PIL import Image
# app
app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'
model = load_model('model1')
disease_map = ['Melanocytic nevi (nv)', 'Melanoma (mel)', 'Benign keratosis-like lesions (bkl)', 'Basal cell carcinoma (bcc)', 'Actinic keratoses (akiec)', 'Vascular lesions (vas)', 'Dermatofibroma (df)']    

# routes
@app.route('/', methods=['GET'])
def home():
    return render_template("index.html")
@app.route('/classify', methods=['POST'])
@cross_origin()
def predict():    
    print("Request received")
    img_name = json.loads(request.data)['img_name']
    img_path = 'C:\\Users\\prati\\python files\\SkinLesionAnalyzer\\' + img_name
    img = np.asarray(Image.open(img_path).resize((32, 32)))
    img = tf.expand_dims(img, axis=0)
    
    preds = model.predict(img)
    dis_idx = preds.argmax()
    op_class = disease_map[dis_idx]
    print(op_class)
    return jsonify({'class': op_class})


if __name__ == '__main__':
    app.run(port=5000, debug=True)
