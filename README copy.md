## The Mission

Create a machine learning model to predict prices on Belgium's real estate sales.

### Installation
Import the following modules : 

´´´
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import matplotlib.ticker as ticker
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score,mean_absolute_error
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
´´´

### Description
The Belgian housing market is influenced by a variety of factors including region, type of property, epcScore and features. This project builds a predictive model to estimate the price of residential real estate. 


#### Step 1 : Data cleaning

1.1 Delete unusefull columns and columns with more than 50% of missing values
1.2 Sorted data by Type for imputation 
    - Mode 'facedeCount' imputation respectivelty for Apartment & House
    - Mode 'epcScore' imputaton respectively for Appartment & House
    - Mode 'buildingCondition' is commun for Appartment & House
1.3 Replace NaN to O and True to 1
1.4 Remove rows with 'price' == 0. 

1.5 Final DataSet : 
´´´
Index(['type', 'subtype', 'bedroomCount', 'bathroomCount', 'province',
       'postCode', 'habitableSurface', 'buildingCondition', 'facedeCount',
       'hasTerrace', 'epcScore', 'price',],
      dtype='object')
´´´

#### Step 2: Analyses Data

2.1 : Check correlation of some variables : 
        - very low correlation price / Surface
        - very low correlation price / bedroomCOunt
        - p-value < 0.05 for price & Province, Subtype, etc... 
    Conclusion : no significant correlation with numeric value - linear regression may not constistant... 
    => Feeling that the categories such as localisation, epcScore,buildingCondition,... have more effect on the price. 


#### Step 3: Model selection

3.1 Load cleaned dataset as df
3.2 Transform dataType : df.get_dummies ['type','province', 'epcScore','subtype','buildingCondition']
3.3 Define varaibles 
    X = df.drop('price', axis=1)
    y = df['price']
3.4 Split Train_Test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

3.5 Try model : 
    3.5.1 Lasso (alpha=1.0): 
            R²: 0.29
            RMSE: 414727.4
            MAE: 198769.38
            
    3.5.2 RandomForest 
            Parameters : n_estimators=300, random_state=42
            R²: 0.75
            RMSE: 247418.24
            MAE: 105631.81
    
    3.5.3 GradientBoostRegressor
            Parameters : n_estimators=2000, learning_rate=0.1, max_depth=4
            R² score : 0.70
            RMSE: 276427.98
            MAE: 111019.29
            
3.6 Next step
    3.6.1 : Cross-validation 
    3.6.2 : Analyses Model
    3.6.3 : Fine tune model and Try with ID colums
    
    


