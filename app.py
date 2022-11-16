from flask import Flask,render_template,request,jsonify
import pickle
import numpy as np
app = Flask(__name__)
model = pickle.load(open("rf_clf_model.pkl", "rb"))

@app.route("/")
def hello_world():
    return render_template("home.html")

@app.route("/predict",methods= ["POST"])
def predict():
    ip_features = [float(x) for x in request.form.values()]
    features = [np.array(ip_features)]
    #features = np.append(features,[0,0,0])
    prediction = model.predict(features)
    #i=[]
    if(prediction==0):
        i ="Setosa"
    elif prediction == 1:
        i = "Verginica"
    elif prediction == 2:
        i = "vernicolor"
    else:
        i = "False"
    return render_template("home.html", prediction_text = "predicted flower is {}".format(i))

if __name__ == "__main__":
    app.run(host="0.0.0.0"port=8080, debug = False)