import numpy as np
import pandas as pd
import os
import pickle
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import RobertaTokenizer,RobertaModel

#load model details from the excel file
df = pd.read_csv('data/Fake Review Detection CDD All Summary.csv')

# load used datasets to create tfidf vectorizer that used during the training
frd_df = pd.read_csv('data/full_prep_data_frd.csv')
frd_df['text'] = frd_df['text'].astype('string')

ylp_df = pd.read_csv('data/filtered_date_labelled_yelp.csv')
ylp_df['text'] = ylp_df['text'].astype('string')

# common configurations
folder_path=os.getcwd()
pre_training_n_samples=200

# create a TF-IDF vectorizer object
frd_vectorizer = TfidfVectorizer(ngram_range=(1, 3))
ylp_vectorizer = TfidfVectorizer(ngram_range=(1, 3))

# fit the vectorizer on the column text & transform the column text to numerical features using TF-IDF
X_train_frd= frd_vectorizer.fit_transform(frd_df['text'][:pre_training_n_samples])

# fit the vectorizer on the column text & transform the column text to numerical features using TF-IDF
X_train_ylp= ylp_vectorizer.fit_transform(ylp_df['text'][:pre_training_n_samples])

# Codes related to glove embeddings
glove100d_path = ''.join([folder_path,'/data/','glove.6B.100d.txt'])
embeddings_index = {}
with open(glove100d_path, encoding="utf8") as f:
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs

def generate_glove_embeddings(text):
    words = text.split()
    embeddings = []
    for w in words:
        if w in embeddings_index:
            embeddings.append(embeddings_index[w])
    if len(embeddings) == 0:
        return np.zeros(100)
    else:
        return np.mean(embeddings, axis=0)

# Load the pre-trained RoBERTa tokenizer
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
model = RobertaModel.from_pretrained('roberta-base')

# Define a function to generate embeddings for a given review using RoBERTa
def generate_roberta_embedding(text):
    text = text[:500]
    input_ids = torch.tensor([tokenizer.encode(text, add_special_tokens=True)])
    with torch.no_grad():
        last_hidden_states = model(input_ids)[0]
    return last_hidden_states.mean(dim=1).numpy().squeeze()

#Generate glove embeddings for training data

def dataload(dataset_name, extraction_technique, classifier, text):
     #load model details from the excel file
    filtered_df=df[ (df['Dataset']==dataset_name.upper()) & (df['Feature Extraction'].str.upper()==extraction_technique.upper()) &(df['Classifier']==classifier.upper() ) ]    
    
    # create an empty list to hold the objects
    objects_list = []

    # iterate through the rows
    for index, row in filtered_df.iterrows():
        #get predictions for the model
        model_name = row['Model Name']

        print('model name', model_name)
        # Load the trained model
        model = load_trained_model(model_name=model_name)

        transformed_data = transformation_type_switch_case(case=extraction_technique,dataset_name=dataset_name,text=text)

        y_pred = model.predict(transformed_data)
        is_fake=int(y_pred[0])

        # create a new object using the values in the row
        new_object = {
            'cdd_method': row['Drift Detection Method'],
            'is_fake':is_fake,
            'accuracy':row['Accuracy'],
            'precision':row['Precision'],
            'recall':row['Recall'],
            'f1_score':row['F1 Score'],
            'mse':row['MSE'],
            'drift_count':int(row['Drift Count']),
            'training_time':row['Time Spent'],
            'memory_used':row['Memory Used']
        }
        
        # append the object to the list
        objects_list.append(new_object)

    return objects_list

def load_trained_model(model_name):
    file_path=''.join([folder_path,'/','models','/',model_name])
    with open(file_path,'rb') as file:
        model=pickle.load(file)
        return model

def transformation_type_switch_case(case,dataset_name,text):
    switcher = {
    "tfidf": get_tfidf_transformation(dataset_name=dataset_name,text=text),
    "glove": get_glove_transformation(text=text),
    "roberta": get_roberta_transformation(text=text)

    }
    return switcher.get(case,"Invalid Extraction Technique")

def get_roberta_transformation(text):
     transformed_data= np.array([generate_roberta_embedding(text)]) 
     return transformed_data

def get_glove_transformation(text):
     transformed_data= np.array([generate_glove_embeddings(text)]) 
     return transformed_data

def get_tfidf_transformation(dataset_name,text):
    # Get the vectorizer
    vectorizer = get_vectorizer(dataset_name)

    # Transform the test datat using same vectorizer
    transformed_data = vectorizer.transform([text])

    return transformed_data

def get_vectorizer(dataset_name):
    if dataset_name == "frd":
        return frd_vectorizer
    else:
        return ylp_vectorizer