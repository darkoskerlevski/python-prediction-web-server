from flask import Flask, request, Response
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from imblearn.over_sampling import SMOTENC
from sklearn.preprocessing import OrdinalEncoder
from lime.lime_tabular import LimeTabularExplainer
import pickle
import json

app = Flask(__name__)

attribute_names = [
    "Status of existing checking account",
    "Duration in month",
    "Credit history",
    "Purpose",
    "Credit amount",
    "Savings account/bonds",
    "Present employment since",
    "Installment rate in percentage of disposable income",
    "Personal status and sex",
    "Other debtors / guarantors",
    "Present residence since",
    "Property",
    "Age in years",
    "Other installment plans",
    "Housing",
    "Number of existing credits at this bank",
    "Job",
    "Number of people being liable to provide maintenance for",
    "Telephone",
    "foreign worker",
    "Credit status"
]

sent_attribute_names = [
    "Status",
    "Duration",
    "CreditHistory",
    "Purpose",
    "Amount",
    "SavingsAccount",
    "PresentEmploymentSince",
    "InstallmentRate",
    "PersonalStatusAndSex",
    "OtherDebtorsAndGuarantors",
    "PresentResidenceSince",
    "Property",
    "Age",
    "OtherInstallmentPlans",
    "Housing",
    "NumberOfExistingCredits",
    "Job",
    "NumberOfPeopleLiableForMaintenance",
    "Telephone",
    "IsForeignWorker"
]

@app.route("/")
def home():
    return "Hello, World!"


@app.route('/predict', methods=['GET', 'POST'])
def result():
    logreg_model = None
    with open('model.pkl', 'rb') as f:
        logreg_model = pickle.load(f)
        
    explainer_data = None
    with open('./explainer_data.pkl', 'rb') as f:
        explainer_data = pickle.load(f)
        
    data = []
    for attr in sent_attribute_names:
        data.append(float(request.form[attr]))
        
    df = pd.DataFrame([data], columns=attribute_names[:-1])
    
    prediction_result = logreg_model.predict_proba([df.iloc[0,:]])
    
    lime_explainer = LimeTabularExplainer(explainer_data.values, mode = 'classification', feature_names = explainer_data.columns.values, class_names = ["1","2"], verbose=True, random_state = 42)
    exp = lime_explainer.explain_instance(df.iloc[0:1].values[0], logreg_model.predict_proba)
    
    d = dict()
    for t in exp.as_map()[1]:
        d[attribute_names[t[0]]] = t[1]
        
    response_data = dict()
    response_data["probaClass1"] = prediction_result[0][0]
    response_data["probaClass2"] = prediction_result[0][1]
    response_data["data"] = d
    
    response_data_json = json.dumps(response_data)
    
    return Response(response_data_json, mimetype="application/json") # response to your request.


if __name__ == "__main__":
    app.run(debug=True)