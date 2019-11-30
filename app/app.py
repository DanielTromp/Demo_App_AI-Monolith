import uuid, os, cv2, logging, imutils

import numpy as np
import tensorflow as tf
from flask import Flask, redirect, render_template, request
from flask import send_from_directory, url_for
from keras.applications.imagenet_utils import preprocess_input
from keras.layers import Input, Convolution2D, ZeroPadding2D, MaxPooling2D
from keras.layers import Flatten, Dense, Dropout, Activation
from keras.models import Model, Sequential, model_from_json
from keras.preprocessing import image
from keras.preprocessing.image import load_img, save_img, img_to_array

application = Flask(__name__)
logger = logging.getLogger(__name__)

#-----------------------
enableGenderIcons = True

male_icon = cv2.imread("images/male.jpg")
male_icon = cv2.resize(male_icon, (40, 40))

female_icon = cv2.imread("images/female.jpg")
female_icon = cv2.resize(female_icon, (40, 40))
#-----------------------

face_cascade = cv2.CascadeClassifier('models/haarcascade_frontalface_default.xml')

def preprocess_image(image_path):
    img = load_img(image_path, target_size=(224, 224))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img

def loadVggFaceModel():
    model = Sequential()
    model.add(ZeroPadding2D((1,1),input_shape=(224,224, 3)))
    model.add(Convolution2D(64, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(Convolution2D(4096, (7, 7), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Convolution2D(4096, (1, 1), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Convolution2D(2622, (1, 1)))
    model.add(Flatten())
    model.add(Activation('softmax'))
    
    return model

def ageModel():
    model = loadVggFaceModel()
    
    base_model_output = Sequential()
    base_model_output = Convolution2D(101, (1, 1), name='predictions')(model.layers[-4].output)
    base_model_output = Flatten()(base_model_output)
    base_model_output = Activation('softmax')(base_model_output)
    
    age_model = Model(inputs=model.input, outputs=base_model_output)
    age_model.load_weights("models/age_model_weights.h5")
    
    return age_model

def genderModel():
    model = loadVggFaceModel()
    
    base_model_output = Sequential()
    base_model_output = Convolution2D(2, (1, 1), name='predictions')(model.layers[-4].output)
    base_model_output = Flatten()(base_model_output)
    base_model_output = Activation('softmax')(base_model_output)

    gender_model = Model(inputs=model.input, outputs=base_model_output)
    gender_model.load_weights("models/gender_model_weights.h5")
    
    return gender_model

def emotionModel():
    emotion_model = model_from_json(
        open("models/facial_expression_model_structure.json", "r").read())
    emotion_model.load_weights("models/facial_expression_model_weights.h5") 
    return emotion_model

age_model = ageModel()
gender_model = genderModel()
emotion_model = emotionModel()

emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')

global graph
graph = tf.get_default_graph()

output_indexes = np.array([i for i in range(0, 101)])

@application.route('/')
def index():
    return render_template('index.html')

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ['jpg', 'jpeg', 'png']

@application.route('/upload', methods=['POST'])
def upload():
    if request.method == 'POST':
        if not request.files.get('file', None):
            msg = 'the request contains no file'
            logger.error(msg)
            return render_template('exception.html', text=msg)
        
        file = request.files['file']
        if file and not allowed_file(file.filename):
            msg = f'the file {file.filename} has wrong extention'
            logger.error(msg)
            return render_template('exception.html', text=msg)

        # Cleanup #
        filelist = [ f for f in os.listdir('tmp') ]
        for f in filelist:
            os.remove(os.path.join('tmp', f))
        # Cleanup #

        filename = str(uuid.uuid4()) + "_" + file.filename
        #filename = file.filename
        file.save('tmp/' + filename)

        logger.info(f'the file {filename} has been successfully saved as {filename}')
        return redirect('/process/' + filename)

@application.route('/process/<filename>')
def task_processing(filename):
    # OpenCV script #
    #clear_session() #keras
    img = cv2.imread('tmp/' + filename)
    
    maximum = 1000
    (h, w) = img.shape[:2]
    if (h > maximum):
        img = imutils.resize(img, height=maximum)
        (h, w) = img.shape[:2]
    if (w > maximum):
        img = imutils.resize(img, width=maximum)
    
    faces = face_cascade.detectMultiScale(img, 1.3, 5)
    for (x,y,w,h) in faces:
        if w > 50: #ignore small faces
            #mention detected face
            cv2.rectangle(img,(x,y),(x+w,y+h),(33,0,195),2) 
            
            #extract detected face
            detected_face = img[int(y):int(y+h), int(x):int(x+w)] 
            emotion_face = cv2.resize(detected_face, (48, 48))
            emotion_face = cv2.cvtColor(emotion_face, cv2.COLOR_BGR2GRAY)
            emotion_pixels = image.img_to_array(emotion_face)
            emotion_pixels = np.expand_dims(emotion_pixels, axis = 0)
            emotion_pixels /= 255

            try:
                #age gender data set has 40% margin around the face. expand detected face.
                margin = 30
                margin_x = int((w * margin)/100); margin_y = int((h * margin)/100)
                detected_face = img[int(y-margin_y):int(y+h+margin_y), int(x-margin_x):int(x+w+margin_x)]

            except:
                print("detected face has no margin")
            
            try:
                #vgg-face expects inputs (224, 224, 3)
                detected_face = cv2.resize(detected_face, (224, 224))
                img_pixels = image.img_to_array(detected_face)
                img_pixels = np.expand_dims(img_pixels, axis = 0)
                img_pixels /= 255

                #find out age and gender
                with graph.as_default():
                    age_distributions = age_model.predict(img_pixels)
                    gender_distribution = gender_model.predict(img_pixels)[0]
                    emotion_distribution = emotion_model.predict(emotion_pixels)

                apparent_age = str(int(np.floor(np.sum(age_distributions * output_indexes, axis = 1))[0]))
                gender_index = np.argmax(gender_distribution)
                if gender_index == 0: gender = "F"
                else: gender = "M"

                max_index = np.argmax(emotion_distribution[0])
                emotion = emotions[max_index]
            
                #background for age gender declaration
                info_box_color = (33,0,195)#(46,200,255)

                triangle_cnt = np.array( [(x+int(w/2), y), 
                    (x+int(w/2)-20, y-20), (x+int(w/2)+20, y-20)] )
                cv2.drawContours(img, [triangle_cnt], 0, info_box_color, -1)
                cv2.rectangle(img,(x+int(w/2)-50,y-10),(x+int(w/2)+55,y-65),
                    info_box_color,cv2.FILLED)
                
                if enableGenderIcons:
                    if gender == 'M': gender_icon = male_icon
                    else: gender_icon = female_icon
                    
                    img[y-60:y-60+male_icon.shape[0], x+int(w/2)-50:x+int(w/2)-50+male_icon.shape[1]] = gender_icon
                else:
                    cv2.putText(img, gender, (x+int(w/2)-42, y - 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

                #labels for age and gender
                cv2.putText(img, apparent_age, (x+int(w/2), y - 35), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(img, emotion, (x+(int(w/2)-15), y - 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            except Exception as e:
                print("exception",str(e))

    cv2.imwrite('tmp/' + filename, img)

    return render_template("result.html", image_name=filename)

@application.route('/result/<filename>')
def send_image(filename):
    return send_from_directory('', 'tmp/' + filename)

if __name__ == '__main__':
    application.run(host='0.0.0.0', port=5000)