import streamlit as st
import pickle
import pandas as pd
import numpy as np
import joblib

#import model
model = joblib.load('xgboost.pkl')
one_hot_encoder = joblib.load('one_hot_encoder.pkl')

#class untuk data form
class form_handler:
    def __init__(self,data_dict):
        self.data_dict = data_dict
        self.dataframe = None

    #konversi ke dataframe pandas
    def convert_to_pandas(self):
        self.dataframe = pd.DataFrame(self.data_dict)
    
    #membersihkan data sebelum diprediksi
    def cleaning_data(self):
        self.dataframe = self.dataframe.replace({
            'Gender':{'🚹 Male':1,'🚺 Female':0},
            'IsActiveMember':{'✅ Yes':1,'❌ No':0}
        })

        result = one_hot_encoder.transform(self.dataframe[['Geography']])
        self.dataframe = pd.concat([self.dataframe,result],axis = 1).drop(columns=['Geography'])
    
    #prediksi data 
    def predict(self):
        x = self.dataframe.values
        prediction_res = model.predict_proba(x).ravel()
        yes_proba = prediction_res[1]
        no_proba = prediction_res[0]
        if yes_proba > no_proba : 
            return ['churned',round(yes_proba*100,2)]
        return ['not churned',round(no_proba*100,2)]

#isi main page   
def main():
    st.header("🏦 Bank Customer Churn Prediction")
    st.write('Predict whether a customer will change a bank')

    #form untuk mengisi data
    with st.form("User Prediction"):
        st.write('**👤 Customer Information**')

        surname = st.text_input("Name",placeholder="Insert the customer name...")
        gender = st.radio('Gender',['🚹 Male','🚺 Female'])
        age = st.number_input('Age',0,100)
        active_member = st.radio('Active Member ?',['✅ Yes','❌ No'])
        credit_score = st.number_input('Credit Score (300 - 850)',300,850)
        balance = st.number_input('Account Balance ($)',0,9999999)
        num_products = st.slider('Number of Used Products',1,4)
        geography = st.radio('Nationality',['Spain','Germany','France'])

        submitted = st.form_submit_button("✨ Predict")

        if submitted:

            #mengumpulkan data untuk diolah menjadi prediksi 
            data = {
                'CreditScore' : [int(credit_score)],
                'Gender' :[gender] , 
                'Age' : [int(age)], 
                'Balance' : [int(balance)], 
                'NumOfProducts': [int(num_products)], 
                'IsActiveMember': [active_member],
                'Geography':[geography]
            }

            submitted_data = form_handler(data)
            submitted_data.convert_to_pandas()
            submitted_data.cleaning_data()
            result,percentage = submitted_data.predict()

            #hasil prediksi
            st.write('**Result :**')
            color = 'red' if result == 'churned' else 'green'
            emoji =  '😟' if result == 'churned' else '😀'
            st.write(f'Your customer, {surname} is predicted :{color}[**{result}** ({percentage}%)] {emoji}')
    

main()
            







