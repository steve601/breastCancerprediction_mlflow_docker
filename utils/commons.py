import os
import sys
import numpy as np
import pickle
from sklearn.metrics import accuracy_score
import pandas as pd
from source.exception import UserException

def save_object(file_path,obj):
    try:
        dir_path = os.path.dirname(file_path)
        
        os.makedirs(dir_path,exist_ok=True)
        
        with open (file_path,'wb') as file_obj:
            pickle.dump(obj,file_obj)
            
    except Exception as e:
        raise UserException(e,sys)
    
def evaluate_model(x_train,y_train,x_test,y_test,models):
    try:
        report = {}
        for i in range(len(list(models))):
            model = list(models.values())[i]
            model.fit(x_train,y_train)
            prediction = model.predict(x_test)
            score = accuracy_score(y_test,prediction)
            report[list(models.keys())[i]] = score
            
        return report
    except:
        pass
    
def load_object(file_path):
    try:
        with open(file_path, 'rb') as file:
            loaded_model = pickle.load(file)
        return loaded_model
    except Exception as e:
        raise UserException(e,sys)