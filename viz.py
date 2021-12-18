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

def actual_vs_predicted(y_train_scaled):
    '''
    Funtion to create a histogram for the mean tax value contrasted against all actual tax values.
    
    '''
    # plot to visualize actual vs predicted. 
    plt.hist(y_train_scaled.tax_value_pred_mean, color='red', alpha=.5,  label="Predicted Tax Values - Mean")
    plt.hist(y_train_scaled.tax_value, color='blue', alpha=.5, label="Actual Tax Values")
    #plt.hist(y_train.G3_pred_median, bins=1, color='orange', alpha=.5, label="Predicted Final Grades - Median")
    plt.xlabel("Tax Value")
    plt.ylabel("Number of Properties")
    plt.legend()
    plt.show()


def age_by_county(train):
    '''
    Function to show a seaborn histogram of the age of homes by county.
    
    '''

    plt.figure(figsize = (16,3))
    plt.subplot(1,3, 1)

    # Title with column name.
    plt.title('LA County')
    # Display histogram for column.
    #plt.boxplot(train[col])
    sns.histplot(data=train[train.county=='LA'].age)
    # Hide gridlines.

    plt.subplot(1,3, 2)
    # Title with column name.
    plt.title('Orange County')
    # Display histogram for column.
    #plt.boxplot(train[col])
    sns.histplot(data=train[train.county=='Orange'].age)
    # Hide gridlines.

    plt.subplot(1,3, 3)
    # Title with column name.
    plt.title('Ventura County')
    # Display histogram for column.
    #plt.boxplot(train[col])
    sns.histplot(data=train[train.county=='Ventura'].age)
    # Hide gridlines.

    plt.grid(False)
    plt.tight_layout()


def histograms1(zillow):

    '''
    Function to create univariate histograms for all continuous features in the zillow data set.
    
    '''
    # Here, we create a for loop that makes a histogram for every column. This is the start of my univariate analysis

    plt.figure(figsize=(16, 3))

    # List of columns

    cols = ['bathroomcnt', 'bedroomcnt', 'calculatedfinishedsquarefeet',
       'fireplacecnt',
       'fullbathcnt', 'garagecarcnt','lotsizesquarefeet']
    # col list from previous analysis...
    # cols = ['bedrooms', 'bathrooms','sq_ft','tax_value', 'age', 'sq_ft_per_bathroom']
    # Note the enumerate code, which is functioning to make a counter for use in successive plots.

    for i, col in enumerate(cols):
        
        # i starts at 0, but plot nos should start at 1
        plot_number = i + 1 
        
        # Create subplot.
        plt.subplot(1,7, plot_number)
        
        # Title with column name.
        plt.title(col)
        
        # Display histogram for column.
        zillow[col].hist(bins=10, edgecolor='black')
        
        # Hide gridlines.
        plt.grid(False)
        
        plt.tight_layout(),

    plt.figure(figsize=(16, 3))

    for i, col in enumerate(cols):
        
        # i starts at 0, but plot nos should start at 1
        plot_number = i + 1 
        
        # Create subplot.
        plt.subplot(1,7, plot_number)
        
        # Title with column name.
        plt.title(col)
        
        # Display histogram for column.
        plt.boxplot(zillow[col])
        
        # Hide gridlines.
        plt.grid(False)
        
        plt.tight_layout()


def histograms2(zillow):

    '''
    Function to create univariate histograms for all continuous features in the zillow data set.
    
    '''
    # Here, we create a for loop that makes a histogram for every column. This is the start of my univariate analysis

    plt.figure(figsize=(16, 3))

    # List of columns

    cols = ['poolcnt', 'roomcnt','yearbuilt',
       'numberofstories', 
       'taxvaluedollarcnt', 'logerror']
    # col list from previous analysis...
    # cols = ['bedrooms', 'bathrooms','sq_ft','tax_value', 'age', 'sq_ft_per_bathroom']
    # Note the enumerate code, which is functioning to make a counter for use in successive plots.

    for i, col in enumerate(cols):
        
        # i starts at 0, but plot nos should start at 1
        plot_number = i + 1 
        
        # Create subplot.
        plt.subplot(1,6, plot_number)
        
        # Title with column name.
        plt.title(col)
        
        # Display histogram for column.
        zillow[col].hist(bins=10, edgecolor='black')
        
        # Hide gridlines.
        plt.grid(False)
        
        plt.tight_layout(),

    plt.figure(figsize=(16, 3))

    for i, col in enumerate(cols):
        
        # i starts at 0, but plot nos should start at 1
        plot_number = i + 1 
        
        # Create subplot.
        plt.subplot(1,6, plot_number)
        
        # Title with column name.
        plt.title(col)
        
        # Display histogram for column.
        plt.boxplot(zillow[col])
        
        # Hide gridlines.
        plt.grid(False)
        
        plt.tight_layout()


def three_histograms(train):
    '''
    Function to draw three histograms for my exploration; please refer to the code and graphic for the variables used.
    
    '''
    # Creating a figure with 3 subplots
    plt.figure(figsize = (16,3))
    plt.subplot(1,3, 1)
    # Here, we see a seaborn barplot of the average tax value in each county
    sns.barplot(x=train.county, y=train.tax_value)
    # And tax value for bedroom count
    plt.subplot(1,3, 2)
    sns.barplot(x=train.bedrooms, y=train.tax_value)
    # Finally tax value for bedroom count
    plt.subplot(1,3, 3)
    sns.barplot(x=train.bathrooms, y=train.tax_value)