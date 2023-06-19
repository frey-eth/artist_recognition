from keras_facenet import FaceNet
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import cv2 # opencv
from mtcnn.mtcnn import MTCNN
from matplotlib import pyplot as plt
from keras.models import load_model
from PIL import Image
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVC

detector = MTCNN()
def get_embedding(model, face):
    # scale pixel values
    face = face.astype('float32')
    sample = np.expand_dims(face, axis=0)
    yhat = model.embeddings(sample)
    return yhat[0]

def get_single_face(img,bbox):
      x1, y1, width, height = bbox
      # deal with negative pixel index
      x1, y1 = abs(x1), abs(y1)
      x2, y2 = x1 + width, y1 + height
      # extract the face
      return img[y1:y2, x1:x2]


def main():
    facenet_model = FaceNet()
    #normalize input vectors
    in_encoder = Normalizer()
    data_emd =  np.load('./7-celebrity-faces-embeddings.npz')
    emdTrainX, trainy, emdTestX, testy = data_emd['arr_0'], data_emd['arr_1'], data_emd['arr_2'], data_emd['arr_3']
    emdTrainX_norm = in_encoder.transform(emdTrainX)
    emdTestX_norm = in_encoder.transform(emdTestX)
    # label encode targets
    out_encoder = LabelEncoder()
    out_encoder.fit(trainy)
    trainy_enc = out_encoder.transform(trainy)
    testy_enc = out_encoder.transform(testy)
    # fit model
    model = SVC(kernel='linear', probability=True)
    model.fit(emdTrainX_norm, trainy_enc)
    # predict
    yhat_train = model.predict(emdTrainX_norm)
    yhat_test = model.predict(emdTestX_norm)

    def predict_img(img,model = model):
        faces_bboxes = detector.detect_faces(img)
        for bbox in faces_bboxes:
            face = get_single_face(img,bbox['box'])
            x, y, width, height = bbox['box']
            face_emd = get_embedding(facenet_model,face)
            samples = np.expand_dims(face_emd, axis=0)
            yhat_class = model.predict(samples)
            yhat_prob = model.predict_proba(samples)
        # get name
            class_index = yhat_class[0]
            class_probability = yhat_prob[0,class_index] * 100
            predict_names = out_encoder.inverse_transform(yhat_class)
            all_names = out_encoder.inverse_transform([0,1,2,3,4,5,6])
            predicted  = 'unknow'
            if yhat_prob[0][yhat_prob[0].argmax()] > 0.6:
                prob  =  yhat_prob[0][yhat_prob[0].argmax()]
                predicted = all_names[yhat_prob[0].argmax()]+ ' ' + str(round(prob,2))
            img = cv2.rectangle(img, (x, y), (x+width, y+height), (0, 155, 255), 2)
            cv2.putText(
                    img,
                    predicted,
                    (int(x), int(y) - 10),
                    fontFace = cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale = 0.6,
                    color = (255, 0, 0),
                    thickness=2
                )
        return img
    
    import gradio as gr
    headline = 'celebrity recognition'
    ui = gr.Interface(fn=predict_img,inputs='image',outputs='image',title=headline)
    ui.launch(share=True)



if __name__ == "__main__":
    main()