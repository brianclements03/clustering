import pandas as pd
import numpy as np
import os
from env import host, user, password
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pydataset import data
import statistics
import seaborn as sns
import env
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
import scipy
from scipy import stats

from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
import sklearn.linear_model
import sklearn.preprocessing
import warnings
warnings.filterwarnings("ignore")
# importing my personal wrangle module
import wrangle


def get_connection(db, user=user, host=host, password=password):
    '''
    This function uses my info from my env file to
    create a connection url to access the Codeup db.
    It takes in a string name of a database as an argument.
    '''
    return f'mysql+pymysql://{user}:{password}@{host}/{db}'
    
    
    
def new_zillow_data():
    '''
    This function reads the zillow data from the Codeup db into a df,
    write it to a csv file, and returns the df.
    '''
    # Create SQL query.

    # I checked, and the following query should and better return exactly 52,442 rows
    sql_query = 'SELECT \
            bedroomcnt, bathroomcnt, calculatedfinishedsquarefeet, taxvaluedollarcnt, yearbuilt, taxamount, fips\
            FROM zillow.properties_2017 AS zprop\
            JOIN zillow.predictions_2017 as zpred USING (parcelid)\
            JOIN zillow.propertylandusetype AS plt USING (propertylandusetypeid)\
            WHERE plt.propertylandusetypeid = 261 OR 279 AND zpred.transactiondate < 2018-01-01;'

    # Read in DataFrame from Codeup db.
    df = pd.read_sql(sql_query, get_connection('zillow'))
    
    return df



def get_zillow_data():
    '''
    This function reads in the zillow data from Codeup database, writes data to
    a csv file if a local file does not exist, and returns a df.
    '''
    if os.path.isfile('zillow_df.csv'):
        
        # If csv file exists, read in data from csv file.
        df = pd.read_csv('zillow_df.csv', index_col=0)
        
    else:
        
        # Read fresh data from db into a DataFrame.
        df = new_zillow_data()
        
        # Write DataFrame to a csv file.
        df.to_csv('zillow_df.csv')
        
    return df

def remove_outliers(df, k, col_list):
    ''' 
    
    Here, we remove outliers from a list of columns in a dataframe and return that dataframe
    
    '''
    
    for col in col_list:

        q1, q3 = df[col].quantile([.25, .75])  # get quartiles
        
        iqr = q3 - q1   # calculate interquartile range
        
        upper_bound = q3 + k * iqr   # get upper bound
        lower_bound = q1 - k * iqr   # get lower bound

        # return dataframe without outliers
        
        df = df[(df[col] > lower_bound) & (df[col] < upper_bound)]
        
    return df

def clean_and_prep_data(df):
    '''
    This function will do some light cleaning and minimal other manipulation of the zillow data set, 
    to get it ready to be split in the next step.
    
    '''
        # renaming columns for ease of reading
    df = df.rename(columns={'bedroomcnt':'bedrooms','bathroomcnt':'bathrooms','calculatedfinishedsquarefeet':'sq_ft','taxvaluedollarcnt':'tax_value','taxamount':'tax_amount','fips':'county'})
    cols = ['bedrooms', 'bathrooms', 'sq_ft', 'tax_value', 'tax_amount']
    # replace fips codes with county names
    df.county = df.county.replace({6037.0:'LA',6059.0: 'Orange',6111.0:'Ventura'})
    # dropping nulls
    df = df.dropna()
    #removing outliers--see the function elsewhere in this file
    df = remove_outliers(df, 1.5, cols)
    # accessing datetime info so as to create an 'age' variable
    from datetime import date
    df.yearbuilt =  df.yearbuilt.astype(int)
    year = date.today().year
    df['age'] = year - df.yearbuilt
    # making a feature called bathrooms_per_sq_ft
    df['sq_ft_per_bathroom'] = df.sq_ft / df.bathrooms
    # dropping the 'yearbuilt' column now that i have the age
    df = df.drop(columns=['yearbuilt', 'tax_amount'])
# Missing values: there were only something around 200 missing values in the data; thus, I have dropped them 
#  due to their relative scarcity.  By removing outliers, several thousand rows were dropped.
    return df


def split_zillow(df):
    '''
    Takes in the zillow dataframe and returns train, validate, test subset dataframes
    '''
    # SPLIT
    # Test set is .2 of original dataframe
    train, test = train_test_split(df, test_size = .2, random_state=123)#, stratify= df.tax_value)
    # The remainder is here divided .7 to train and .3 to validate
    train, validate = train_test_split(train, test_size=.3, random_state=123)#, stratify= train.tax_value)

    # return train, validate, test

    X_train = train.drop(columns=['tax_value'])
    y_train = pd.DataFrame(train.tax_value, columns=['tax_value'])

    X_validate = validate.drop(columns=['tax_value'])
    y_validate = pd.DataFrame(validate.tax_value, columns=['tax_value'])

    X_test = test.drop(columns=['tax_value'])
    y_test = pd.DataFrame(test.tax_value, columns=['tax_value'])

    return train, validate, test, X_train, y_train, X_validate, y_validate, X_test, y_test


def encode_zillow(df):
    '''
    This is encoding a few of the zillow columns for later modelling; it drops the original column 
    once it has been encoded
    
    '''
    # ordinal encoder? sklearn.OrdinalEncoder

    cols_to_dummy = df['county']
    dummy_df = pd.get_dummies(cols_to_dummy, dummy_na=False, drop_first=False)
    df = pd.concat([df, dummy_df], axis = 1)
    #df.columns = df.columns.astype(str)
    # I ended up renaming counties in an above function; the other encoded cols are renamed here:
    #df.rename(columns={'6037.0':'LA', '6059.0': 'Orange', '6111.0':'Ventura'}, inplace=True)
    # I have commented out the following code bc i think i might want to have the county column for exploration
    #df = df.drop(columns='county')
    return df



def scale_zillow(train, validate, test):
    '''
    Takes in the zillow dataframe and returns SCALED train, validate, test subset dataframes
    '''
    # SCALE
    # 1. create the object
    scaler = sklearn.preprocessing.MinMaxScaler()
    # 2. fit the object
    scaler.fit(train[['bedrooms', 'bathrooms', 'sq_ft', 'tax_value', 'age','sq_ft_per_bathroom',
       'LA', 'Orange', 'Ventura']])
    # 3. use the object. Scale all columns for now
    train_scaled =  scaler.transform(train[['bedrooms', 'bathrooms', 'sq_ft', 'tax_value', 'age','sq_ft_per_bathroom',
       'LA', 'Orange', 'Ventura']])
    train_scaled = pd.DataFrame(train_scaled, columns=['bedrooms', 'bathrooms', 'sq_ft', 'tax_value', 'age','sq_ft_per_bathroom',
       'LA', 'Orange', 'Ventura'])

    validate_scaled =  scaler.transform(validate[['bedrooms', 'bathrooms', 'sq_ft', 'tax_value', 'age','sq_ft_per_bathroom',
       'LA', 'Orange', 'Ventura']])
    validate_scaled = pd.DataFrame(validate_scaled, columns=['bedrooms', 'bathrooms', 'sq_ft', 'tax_value', 'age','sq_ft_per_bathroom',
       'LA', 'Orange', 'Ventura'])

    test_scaled =  scaler.transform(test[['bedrooms', 'bathrooms', 'sq_ft', 'tax_value', 'age','sq_ft_per_bathroom',
       'LA', 'Orange', 'Ventura']])
    test_scaled = pd.DataFrame(test_scaled, columns=['bedrooms', 'bathrooms', 'sq_ft', 'tax_value', 'age','sq_ft_per_bathroom',
       'LA', 'Orange', 'Ventura'])

    # 4. Divide into x/y

    X_train_scaled = train_scaled.drop(columns=['tax_value'])
    y_train_scaled = pd.DataFrame(train_scaled.tax_value, columns=['tax_value'])

    X_validate_scaled = validate_scaled.drop(columns=['tax_value'])
    y_validate_scaled = pd.DataFrame(validate_scaled.tax_value, columns=['tax_value'])

    X_test_scaled = test_scaled.drop(columns=['tax_value'])
    y_test_scaled = pd.DataFrame(test_scaled.tax_value, columns=['tax_value'])

    return train_scaled, X_train_scaled, y_train_scaled, validate_scaled, X_validate_scaled, y_validate_scaled, test_scaled, X_test_scaled, y_test_scaled