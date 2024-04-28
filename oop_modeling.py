import pandas as pd
import numpy as np 
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import ADASYN
from sklearn.metrics import classification_report
import xgboost as xgb
import pickle

#kelas untuk dataset
class dataset:
    def __init__(self,file_source:str):
        self.file_source = file_source
        self.dataframe = pd.read_csv(self.file_source)
        self.input_column = None 
        self.output_column = None

    #menghapus kolom dalam dataset
    def drop_columns(self,columns:list):
        self.dataframe = self.dataframe.drop(columns=columns)

    #inisiasi input dan output dalam dataset
    def set_input_output_column(self,input_column:list,output_column:list):
        self.output_column = self.dataframe[output_column]
        self.input_column = self.dataframe[input_column]
    
    #mendapatkan mean dari kolom dataset
    def get_mean(self,column):
        return self.dataframe[column].mean()

    #memisahkan data menjadi train dan test
    def split_data(self, test_proportion:float):
        x_train,x_test,y_train,y_test = train_test_split(self.input_column,self.output_column,test_size=test_proportion)
        return [x_train,y_train],[x_test,y_test]

#kelas untuk model 
class model_handler:
    def __init__(self,train_data:list,test_data:list):
        self.x_train,self.y_train = train_data
        self.x_test,self.y_test = test_data
        self.model = xgb.XGBClassifier()
        self.oneHotEncoder =  OneHotEncoder(sparse_output=False).set_output(transform='pandas')
        self.adaysn = ADASYN()

    #mengisi data kosong
    def fill_na(self,column,alt_value):
        self.x_train[column] = self.x_train[column].fillna(alt_value)
        self.x_test[column] = self.x_test[column].fillna(alt_value)
    
    #feature encoding data kategori
    def encode_categorical_data(self,encode_dict:dict):
        self.x_train = self.x_train.replace(encode_dict)
        self.x_test = self.x_test.replace(encode_dict)

    #one hot encoder data kategori
    def one_hot_encoder(self,column):
        result_train = self.oneHotEncoder.fit_transform(self.x_train[[column]])
        result_test = self.oneHotEncoder.transform(self.x_test[[column]])

        self.x_train = pd.concat([self.x_train,result_train],axis = 1).drop(columns=[column])
        self.x_test = pd.concat([self.x_test,result_test],axis = 1).drop(columns=[column])

    #melakukan oversampling dengan metode ADASYN
    def oversampling_adaysn(self):
        self.x_train, self.y_train = self.adaysn.fit_resample(self.x_train, self.y_train)

    #melatih model 
    def fit(self):
        self.model.fit(self.x_train,self.y_train)
    
    #mengevaluasi model 
    def evaluate(self):
        y_pred = self.model.predict(self.x_test)
        print(classification_report(self.y_test,y_pred))
    
    #mengkonversi model menjadi pickle
    def convert_to_pickle(self):
        objects = {
            'xgboost':['xgboost.pkl',self.model],
            'one_hot_encoder':['one_hot_encoder.pkl',self.oneHotEncoder]
        }
        
        for obj in objects:
            name_file = objects[obj][0]
            model = objects[obj][1]
            pickle.dump(model, open(name_file, 'wb'))


dataset_1 = dataset('data_C.csv')
dataset_1.drop_columns(['Unnamed: 0','CustomerId','Surname'])

input_cols = ['CreditScore', 'Gender', 'Age', 'Balance', 'NumOfProducts', 'IsActiveMember','Geography']
output_cols = ['churn']

dataset_1.set_input_output_column(input_cols,output_cols)
train,test = dataset_1.split_data(0.15)

model_handler_1 = model_handler(train,test)
model_handler_1.fill_na('CreditScore',dataset_1.get_mean('CreditScore'))
model_handler_1.encode_categorical_data({'Gender':{'Male':1,'Female':0}})
model_handler_1.one_hot_encoder('Geography')
model_handler_1.oversampling_adaysn()
model_handler_1.fit()
model_handler_1.evaluate()
model_handler_1.convert_to_pickle()