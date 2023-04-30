import os
import numpy as np
import cv2
from PIL import Image
from keras.models import load_model
from flask import Flask, request, render_template, redirect, flash, abort
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.secret_key = 'super_secret_key'

ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/", methods=['GET'])
def login():
    return render_template("login.html")

@app.route("/verify", methods=['POST'])
def loggedIn():
    userName = request.form.get('userName')
    password = request.form.get('password')
    if userName == 'admin' and password == 'password':
        return redirect('/test')
    else:
        return render_template('404NotFound.html')

model = load_model('G:/deep learning project/deep learning project/Brain Tumor Image Classification/BrainTumor10EpochsNew.h5')

def get_className(classNo):
    if classNo == 0:
        return "No brain tumor is detected"
    elif classNo == 1:
        return "Yes, brain tumor is detected"

def get_prediction(img):
    image = cv2.imread(img)
    image = Image.fromarray(image, 'RGB')
    image = image.resize((64, 64))
    image = np.array(image)
    input_img = np.expand_dims(image, axis=0)
    prediction = model.predict(input_img)
    return prediction

@app.route('/test', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            prediction = get_prediction(file_path)
            result = get_className(np.argmax(prediction))
            return result
        else:
           return "Invalid image extension, please upload only JPG, JPEG, or PNG files"

@app.errorhandler(404)
def page_not_found(e):
    return render_template('404NotFound.html'), 404

if __name__ == '__main__':
    app.config['UPLOAD_FOLDER'] = 'uploads/'
    app.run(debug=True, port=8000)
