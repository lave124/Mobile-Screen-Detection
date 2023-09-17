import os
import cv2
import tensorflow
import numpy as np
from werkzeug.utils import secure_filename
from flask import Flask, request, render_template, url_for, send_from_directory
import numpy as np
from skimage.color import rgb2gray
from skimage.io import imread
from skimage import feature, img_as_bool
from skimage.morphology import binary_dilation, binary_erosion
import os
import cv2
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
crack_prob=0

app = Flask(__name__)
model = tensorflow.keras.models.load_model('vgg mobile2.h5')
model1=tensorflow.keras.models.load_model('vgg model2.h5')
classes = ["Cracked", "Not Cracked"]
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
MOBILE_TEMPLATE_PATH = 'mobile_template.png'

class_labels=["Mobile","No Mobile"]

def preprocess(url):
    img = imread(url)
    img = rgb2gray(img)
    img_edge = binary_erosion(binary_dilation(feature.canny(img, sigma =.1)))
    return img, img_edge

def edge_prob(window, cut_off):
    pixels  = np.array(window.ravel())
    if ((np.count_nonzero(pixels)/len(pixels))>cut_off):
        return 1
    else:
        return 0


def sliding_mat(img,window_x=10,window_y=10, cut_off=0.1):
    
    arr_x = np.arange(0,img.shape[0],window_x)
    arr_y = np.arange(0,img.shape[1],window_y)

    A = np.zeros((len(arr_x),len(arr_y)))

    for i,x in enumerate(arr_x):
        for j,y in enumerate(arr_y):
            window = img[x:x+window_x,y:y+window_y]
            A[i,j] = edge_prob(window, cut_off=cut_off)
    
    return A, arr_x, arr_y

def is_mobile_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (224, 224))
    img = np.expand_dims(img, axis=0)
    predictions1 = model1.predict(img)
    predicted_class1 = class_labels[np.argmax(predictions1)]
    if predicted_class1=="Mobile":

        return True
    
    else:
        return False



def preprocess_image(image_path):
    if not is_mobile_image(image_path):
        raise ValueError(''' Image You are Providing Is Not of a Mobile IF ITS A MOBILE
         CHOOSE THE DIFFRENT IMAGE FACING Straight ''')
    else:
        img = cv2.imread(image_path)
        img = cv2.resize(img, (224, 224))
        img = tensorflow.keras.applications.mobilenet_v2.preprocess_input(img)
        img = np.expand_dims(img, axis=0)
        return img


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST','GET'])
def predict():
    file = request.files['image']
    if not file:
        return render_template('index.html')
    
    try:
        # Save the uploaded file
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        Category=""

        # Check if the image is of a mobile
        if is_mobile_image(file_path):

            Category+="Mobile"
            
            
            # Preprocess the image
            img = preprocess_image(file_path)

            # Make a prediction using the model
            prediction = model.predict(img,batch_size=1)

            # Get the predicted class label
            label = classes[np.argmax(prediction)]
            if label=="Cracked":
                img2,c2 = preprocess(file_path)
                A, arr_x, arr_y = sliding_mat(c2, window_x=10, window_y=10, cut_off=0.1)
                crack_prob = np.sum(A) / A.size
                decision=""
                if crack_prob>0.7:
                    decision+="Severed Damaged"
                else:
                    decision+="Not too much damaged"
            else:
                label=label
                decision=" Not Damaged"
                
        else:
            label = "Image you are providing is not a mobile Please Choose a Image with mobile"
            decision="Can't take any Decision Because its not a mobile"
            Category="Unknown"

        # Generate the URL for the uploaded file
        image_url = url_for('static', filename='uploads/' + filename)

        # Render the prediction result in the HTML template
        return render_template('index.html', label=label,decision=decision,image_url=image_url,Category=Category)

    except ValueError as e:
        error_message = str(e)
        return render_template('index.html', error=error_message)
    


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)




if __name__ == '__main__':
    app.run(debug=True)
