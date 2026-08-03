from flask import Flask,render_template,request
import joblib
import pandas as pd

app=Flask(__name__)
# load model & files
model=joblib.load('model.pkl')
columns=joblib.load('columns.pkl')
dropdown=joblib.load('dropdown_values.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict')
def predict():
    return render_template('predict.html',
                           makes=dropdown['make'],
                           fules=dropdown['fuel-type'],
                           bodys=dropdown['body-style']
                           )
@app.route('/predict_price',methods=['POST'])
def predict_price():
    make = request.form['make']
    fuel=request.form['fuel_type']
    body=request.form['body_style']
    engine_size=float(request.form['engine_size'])

    data={}
    for col in columns:
        data[col]=0
    if "engine-size" in data:
        data['engine-size']=engine_size

        make_col='make_' + make
        fuel_col='fuel-type_' + fuel
        body_col='body-style_' + body

        if make_col in data:
            data[make_col] = 1
        if fuel_col in data:
            data[fuel_col] = 1
        if body_col in data:
            data[body_col] = 1

            df=pd.DataFrame([data])
            price=model.predict(df)[0]
            return render_template('result.html', price=round(price,2),
                                   make=make,
                                   fuel=fuel,
                                   body=body,
                                   engine_size=engine_size)
@app.route('/about')
def about():
     return render_template('about.html')
@app.route('/contact')
def contact():
     return render_template('contact.html')
if __name__ == '__main__':
    app.run(debug=True)
        
    


