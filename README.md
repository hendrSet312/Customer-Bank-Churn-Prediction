# Customer Bank Churn Prediction 
This project aims to classify customer churn based on given customer data. Predicting churn helps banks retain their customers by identifying those who are likely to leave and taking proactive measures. The classification model is built in python and deployed in streamlit. 

## Workflow 

### Dataset 
The dataset is a customer data, which consists of customer profile (geography ,gender, salary and age) and customer bank information (used products information, account balance and active member status ).

### Exploratory Data Analysis 
The modeling started with data analysis to inspect data characteristic and anomalies. 

### Data preprocessing
- removing unnecessary columns
- filling unknown data
- doing Binary encoding for gender column and one hot encoding for geography variable
- Feature selection using f-score
- Oversampling churn variable using ADASYN
  
### Modelling 
- This process evaluated 2 model, random forest and xgboost
- XGboost was chosen, because it had higher accuracy and f1 score rather than random forest
- The best model exported into pickle file format

### Model Pipeline
- Pipeline was built in oop_modeling.py file
- Consists of data preprocessing and modeling pipeline

### Model Deployment
- Building a simple website to show how model works
- User should fill input about their account information and profile
- Inputed data will be used for model prediction
- The model give the result output

## Project Structure 
1. Data_preprocessing.ipynb : Jupyter notebook files for data analysis,data preprocessing and modeling
2. One_hot_encoder.pkl : Pickle file to convert a categorical data into binary column for each category
3. oop_modeling.py : Python file consists of modeling and data preprocessing pipeline using OOP
4. prediction.py : python file to deploy model with streamlit framework
5. requirements.txt : list library used in this project

## Setup Environment 
1. Open Anaconda in CMD
2. Create a virtual environment:
 ```
 conda create --name project_churn python=3.9
 ```
3. Activate the virtual environment:
 ```
 conda activate project_churn
 ```
4. Navigate the virtual environment to the project directory:
 ```
 cd /d <submission directory location>
 ```
 example
 ```
 cd /d \project_churn
 ```
5. Install the required modules:
 ```
 pip install -r requirements.txt
 ```
