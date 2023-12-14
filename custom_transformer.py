from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import pandas as pd

# Missing Value Merchant City and Zip imputation and update to ONLINE, FOREIGN AND LOCAL 
class GenZipCategoryTransformer(BaseEstimator, TransformerMixin):
    def __init__(self,variables=None):
        self.variables = variables
        
    def fit(self, X, y = None):
        return self
    
    def transform(self, X):
        # Update all NaN merchant states to ONLINE if City is Online
        X['Merchant State'] = np.where(X['Merchant City']=='ONLINE','ONLINE',X['Merchant State'])

        # Assign Zip as 100 foreign
        X['Zip'] = np.where((X['Merchant State'].notnull()) & (X['Merchant State']!='ONLINE') & (X['Zip'].isnull()),100,X['Zip'])

        # Assign Zip as 0 for online
        X['Zip'] = np.where(X['Zip'].isnull(),0,X['Zip'])

        # Then make all local as 20
        X['Zip']=np.where(X['Zip']>100,20,X['Zip'])

        X.drop(['Merchant State', 'Merchant City'],axis=1, inplace=True)

        return X


# Missing Value Merchant City and Zip imputation and update to ONLINE, FOREIGN AND LOCAL 
class GenMCCGroupTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, variables=None):
        self.variables = variables
        
    def fit(self, X, y = None):
        return self
    
    def transform(self, X):
        X['MCC_Group'] = X.apply(lambda x: self.MCC_Group(x.MCC), axis=1)
        X.drop(columns=['MCC'], inplace=True)
        return X
    
    def MCC_Group(self, MCC):  # Reference: https://www.citibank.com/tts/solutions/commercial-cards/assets/docs/govt/Merchant-Category-Codes.pdf
        if (MCC >= 1) & (MCC <= 1499):
            return "Agricultural Services"
        elif (MCC >= 1500) & (MCC <= 2999):
            return "Contracted Services"
        elif (MCC >= 4000) & (MCC <= 4799):
            return "Transportation Services"
        elif (MCC >= 4800) & (MCC <= 4999):
            return "Utility Services"
        elif (MCC >= 5000) & (MCC <= 5599):
            return "Retail Outlet Services"
        elif (MCC >= 5600) & (MCC <= 5699):
            return "Clothing Stores"
        elif (MCC >= 5700) & (MCC <= 7299):
            return "Miscellaneous Stores"
        elif (MCC >= 7300) & (MCC <= 7999):
            return "Business Services"
        elif (MCC >= 8000) & (MCC <= 8999):
            return "Professional Svcs and Membership Orgs"
        elif (MCC >= 9000) & (MCC <= 9999):
            return "Government Services"
        elif (MCC >= 3000) & (MCC <= 3299):
            return "Airlines"
        elif (MCC >= 3300) & (MCC <= 3499):
            return "Car Rental"
        elif (MCC >= 3500) & (MCC <= 3999):
            return "Lodging"
        else:
            return "Others"
        
# Get Hour from Time and Change amount to float
class GenHourTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, variables=None):
        self.variables = variables
        
    def fit(self, X, y = None):
        return self
    
    def transform(self, X):
        # Extract the hours and minutes from Time to perform a more refined time series analysis
        X["Hour"] = X["Time"].str[0:2].astype(int)
        X["Minute"] = X["Time"].str[3:5]

        # Drop Time and Minutes 
        X.drop(['Time', 'Minute'],axis=1, inplace=True)

        X["Amount"]=X["Amount"].str.replace("$","").astype(float)

        return X
    

# Missing Value Merchant City and Zip imputation and update to ONLINE, FOREIGN AND LOCAL 
class GenDayOfWeekTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, variables=None):
        self.variables = variables
        
    def fit(self, X, y = None):
        return self
    
    def transform(self, X):
        X['Date'] = pd.to_datetime(X[['Year', 'Month', 'Day']])

        X['Day of Week'] = X['Date'].dt.dayofweek

        X.drop(['Year','Date'],axis=1, inplace=True)
        
        return X


# EncodeDummiesTransformer
class EncodeDummiesTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, variables=None):
        self.variables = variables
        print(f'EncodeDummiesTransformer: {self.variables}')
        
    def fit(self, X, y=None):
        # One-hot encoding for y 
        X_dummies = pd.get_dummies(X[self.variables ], prefix=self.variables)
        self.X_columns= X_dummies.columns
        return self
    
    def transform(self, X, y=None):
        y=self.variables
        
        X_dummies = pd.get_dummies(X[y], prefix=y)
        X_dummies =X_dummies.reindex(columns=self.X_columns, fill_value=0)

        # Concat the new dummy columns into the dataframe
        X = pd.concat([X, X_dummies], axis=1)

        # Drop the categorical column
        X.drop([y],axis=1, inplace=True)

        return X
