import numpy as np
from tensorflow import keras
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50 , preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D 
import os
from tqdm import tqdm
import pickle
from numpy.linalg import norm

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
        
