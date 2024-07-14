from flask import Flask,request,render_template
import pickle
import pandas as pd

app = Flask(__name__)

def load_object(file_path):
    with open(file_path, 'rb') as file:
        loaded_model = pickle.load(file)
    return loaded_model

model_path = 'models/model.pkl'
scaler_path ='models/scaler.pkl'

model = load_object(model_path)
scaler = load_object(scaler_path)

@app.route('/')
def homepage():
    return render_template('index.html')

@app.route('/predict',methods = ['POST'])
def prediction():
    x = [i for i in request.form.values()]
    
    columns = ['mean_radius', 'mean_texture', 'mean_perimeter', 'mean_area','mean_smoothness']
    x = pd.DataFrame([x],columns=columns)
    
    x = scaler.transform(x)
    
    y_pred = model.predict(x)
    msg = 'Patient is likely to have cancer' if y_pred == 1 else 'Patient is unlikely to have cancer'
    
    return render_template('index.html',text=msg)

if __name__ == "__main__":
    app.run(host="0.0.0.0",port=5000)