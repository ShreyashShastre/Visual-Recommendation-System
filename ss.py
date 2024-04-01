import streamlit as st
import os
from PIL import Image
import numpy as np
import pickle
from sklearn.neighbors import NearestNeighbors
from train import extract_features
import cv2
from tensorflow import keras
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50 , preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D 
from tqdm import tqdm
from numpy.linalg import norm

inf = False

cap = cv2.VideoCapture(1)
feature_list = np.array(pickle.load(open('embeddings.pkl','rb')))
filenames = pickle.load(open('filenames.pkl','rb'))





def save_uploaded_file(uploaded_file):
    try:
        with open(os.path.join('uploads',uploaded_file.name),'wb') as f:
            f.write(uploaded_file.getbuffer())
        return 1
    except:
        return 0



def recommend(features,feature_list):
    neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')
    neighbors.fit(feature_list)

    _, indices = neighbors.kneighbors([features])

    return indices
def show_recommend(image):
    features = extract_features(image)
    indices = recommend(features,feature_list)
    _,col2,col3,col4,col5,col6 = st.beta_columns(6)
    with col2:
        st.image(f'images/{filenames[indices[0][1]]}')
    with col3:
        st.image(f'images/{filenames[indices[0][2]]}')
    with col4:
        st.image(f'images/{filenames[indices[0][3]]}')
    with col5:
        st.image(f'images/{filenames[indices[0][4]]}')
    with col6:
        st.image(f'images/{filenames[indices[0][5]]}')
    return True

st.title('Recommendation System')
st.subheader('Upload/capture an image to get recommendations')
capture = st.checkbox('Capture image')


if capture:
    capst = st.button('Start Video')
    st.markdown('---')
    stframe = st.empty()
    while capst:
        ret, frame = cap.read()
        frame2 = frame.copy()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        cv2.imshow('frame',frame2)
        stframe.image(frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.imwrite('capture.jpg',frame)
            cap.release()
            cv2.destroyAllWindows()
            print('Execution completed')
            inf = True
            if  show_recommend('capture.jpg'):
                st.success('Recommendation completed')
            break


else:
    uploaded_file = st.file_uploader("Choose an image")
    if uploaded_file is not None:
        if save_uploaded_file(uploaded_file):
            display_image = Image.open(uploaded_file)
            st.image(display_image)
            col1,col2,col3,col4,col5,col6 = st.beta_columns(6)
            with col2:
                st.image(f'images/{filenames[recommend(extract_features(os.path.join("uploads",uploaded_file.name)),feature_list)[0][1]]}')
            with col3:
                st.image(f'images/{filenames[recommend(extract_features(os.path.join("uploads",uploaded_file.name)),feature_list)[0][2]]}')
            with col4:
                st.image(f'images/{filenames[recommend(extract_features(os.path.join("uploads",uploaded_file.name)),feature_list)[0][3]]}')
            with col5:
                st.image(f'images/{filenames[recommend(extract_features(os.path.join("uploads",uploaded_file.name)),feature_list)[0][4]]}')
            with col6:
                st.image(f'images/{filenames[recommend(extract_features(os.path.join("uploads",uploaded_file.name)),feature_list)[0][5]]}')
        else:
            st.header("Some error occured in file upload")

model = ResNet50(weights='imagenet', include_top=False , input_shape=(224,224,3))
model.trainable = False

model  = tf.keras.models.Sequential([model,GlobalAveragePooling2D()])

def extract_features(img_path,model=model):
    img = image.load_img(img_path,target_size=(224,224))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    result = model.predict(preprocessed_img).flatten()
    normalized_result = result / norm(result)

    return normalized_result


if __name__ == '__main__':

    filenames = []
    print('extracting file names....')
    for file in tqdm(os.listdir('images')):
        filenames.append(file)

    featuresList = []

    for file in tqdm(filenames):
        featuresList.append(extract_features(f'images/{file}',model))

    pickle.dump(featuresList,open('features.pkl','wb'))
    pickle.dump(filenames,open('filenames.pkl','wb'))
        
