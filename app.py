import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
from flask import Flask,request,jsonify,render_template
from custom_transformer import GenZipCategoryTransformer, GenMCCGroupTransformer, GenHourTransformer, GenDayOfWeekTransformer, EncodeDummiesTransformer

application=Flask(__name__)
app=application

# scaler
# scaler = pickle.load(open("scaler.pkl", 'rb'))
data_zip_preprocessing = pickle.load(open("zip_preprocessing.pkl", 'rb'))
data_mcc_preprocessing = pickle.load(open("mcc_preprocessing.pkl", 'rb'))
data_time_preprocessing = pickle.load(open("time_preprocessing.pkl", 'rb'))
data_dayOfWeek_preprocessing = pickle.load(open("dayOfWeek_preprocessing.pkl", 'rb'))
data_dummy_mcc_preprocessing = pickle.load(open("dummy_mcc_preprocessing.pkl", 'rb'))
data_dummy_dayOfWeek_preprocessing = pickle.load(open("dummy_dayOfWeek_preprocessing.pkl", 'rb'))
data_dummy_zip_preprocessing = pickle.load(open("dummy_zip_preprocessing.pkl", 'rb'))
model_algo_dtree = pickle.load(open("cc_fraud_pipeline_1_model.pkl", 'rb'))


## Year, Month, Day, Time, Amount, Merchant City, Merchant State, Zip, MCC

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata', methods=['Get', 'Post'])
def predict_datapoint():
    if request.method=='POST':
      #Year, Month, Day, Time, Amount, Merchant City, MerchantState
        # new_data_sc=standard_scaler.transform([[Temperature,RH,Ws,Rain,FFMC,DMC,ISI,Classes,Region]])
        # result=ridge_model.predict(new_data_sc)
        Year = int(request.form.get('Year'))
        Month = int(request.form.get('Month'))
        Day = int(request.form.get('Day'))
        Time = str(request.form.get('Time'))
        Amount = str(request.form.get('Amount'))
        MerchantCity = str(request.form.get('MerchantCity'))
        MerchantState = str(request.form.get('MerchantState'))
  


        Zip = (request.form.get('Zip'))
        if Zip=='':
            Zip=np.NaN
        else:
            Zip = float(Zip)
   

        MCC = int(request.form.get('MCC'))
   

        if Zip=='':
            Zip=np.NaN
        if MerchantState=='':
            MerchantState=np.NaN

        model_data = pd.DataFrame({
                'Year' : [Year],
                'Month' : [Month],
                'Day' : [Day],
                'Time' : [Time],
                'Amount' : [Amount],
                'Merchant City' : [MerchantCity],
                'Merchant State' : [MerchantState],
                'Zip' : [Zip],
                'MCC' : [MCC]
              
                })

        # Encode categorical data to numeric
        df_sample_new = model_data
        x_new_trsfrm = data_zip_preprocessing.transform(df_sample_new)
        x_new_trsfrm = data_mcc_preprocessing.transform(x_new_trsfrm)
        x_new_trsfrm = data_time_preprocessing.transform(x_new_trsfrm)
        x_new_trsfrm = data_dayOfWeek_preprocessing.transform(x_new_trsfrm)
        x_new_trsfrm = data_dummy_mcc_preprocessing.transform(x_new_trsfrm)
        x_new_trsfrm = data_dummy_dayOfWeek_preprocessing.transform(x_new_trsfrm)
        x_new_trsfrm = data_dummy_zip_preprocessing.transform(x_new_trsfrm)
        
        # Predict target variable
        y_pred_test_dtree = model_algo_dtree.predict(x_new_trsfrm)
        
        # Class prediction    
        if [y_pred_test_dtree][0][0] == 1:
            result = 'This Transaction is labelled as FRAUD.'
        else:
            result = 'This Transaction is labelled as NOT FRAUD'
            
        # return render_template('index.html', result=result[0])
        # age = float(request.form.get('age'))
        return render_template('index.html', result=result)
    
    else:
        return render_template('index.html')


# if __name__=="__main__":
#     app.run(host="0.0.0.0", port=5000)
if __name__ == "__main__":
    app.run(debug=True)