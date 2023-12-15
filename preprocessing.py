import scipy
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt 

import geopy.distance

import warnings

import statsmodels.api as sm
from sklearn import metrics
from scipy import stats
from scipy.stats import norm, skew

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

"""
VISUALIZACION DE OUTLIERS
"""

def outliers1(df):
    """
    Función que estudia los outliers de varias variables,
    tanto unidimensional como multidimensionales.
    """
    # price
    print("Boxplot de price:")
    outliers_1D(df, 'price')
    print("-"*50)
    # yr_built
    print("Boxplot de yr_built:")
    outliers_1D(df, 'yr_built')
    print("-"*50)
    # sqft_above
    print("Boxplot de sqft_above:")
    outliers_1D(df, 'sqft_above')
    print("-"*50)
    # bathrooms
    print("Boxplot de bathrooms:")
    outliers_1D(df, 'bathrooms')
    print("-"*50)

def outliers2(df):
    # Grade~Price
    print("Boxplot de price en funcion de grade:")
    outliers_MD(df, 'grade', 'price')
    print("-"*50)
    # Condition~Price
    print("Boxplot de price en funcion de condition:")
    outliers_MD(df, 'condition', 'price')
    print("-"*50)
    # Floors~Price
    print("Boxplot de price en funcion de floors:")
    outliers_MD(df, 'floors', 'price')
    print("-"*50)

    
def outliers_1D(df, variable):
    sns.boxplot(x=df[variable])
    plt.title("Boxplot de " + variable)
    plt.tight_layout()
    plt.show()
    
def outliers_MD(df, variable_x, variable_y):
    plt.figure(figsize=(15,5))
    sns.boxplot(x=variable_x, y=variable_y, data=df)
    plt.title("Boxplot de " + variable_x + " frente a " + variable_y)
    plt.tight_layout()
    plt.show()
    
    
"""
FEATURE ENGINEERING
"""
   
def feature_eng(df):
    # Relación entre el área construida y el tamaño del terreno
    df['sqft_ratio'] = df['sqft_living'] / df['sqft_lot']
    
    # Edad de la casa en el momento de la venta
    df['age'] = 2015 - df['yr_built']
    
    # Edad desde la última renovación
    
    df['age'] = 2015 - df['yr_renovated']
    
    # One-hot encoding sobre la variable 'condition'
    location_dummies = pd.get_dummies(df['condition'], prefix='condition')
    df = pd.concat([df, location_dummies], axis=1)
    
    # One-hot encoding sobre la variable 'view'
    location_dummies = pd.get_dummies(df['view'], prefix='view')
    df = pd.concat([df, location_dummies], axis=1)
    
    # Frecuencia respecto a la variable 'view'
    view_counts = df['view'].value_counts(normalize=True)
    df['view_freq'] = df['view'].map(view_counts)

    # Coordenadas del centro de la ciudad de Seattle
    seattle_center = (47.6062, -122.3321)
    
    # Crear una nueva característica que represente la distancia de la casa al centro de la ciudad
    distances = []
    for i, row in df.iterrows():
        coords = (row['lat'], row['long'])
        distance = geopy.distance.distance(coords, seattle_center).km
        distances.append(distance)
    df['distance_to_seattle'] = distances

    
    # Mostrar las nuevas características creadas
    display(df.head())
    return df
    
"""
CORRELACIONES
"""
    
def corr(df):
    plt.figure(figsize=(20,10))
    correlaciones= df.corr()
    sns.heatmap(correlaciones,cmap="GnBu",annot=True) 
    correlaciones
    

"""
NORMALIZACION
"""

def grafico_norm(df, variable):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sns.distplot(df[variable],color="red",kde=True,fit=norm)
        
    #Get also the QQ-plot
    fig = plt.figure()
    res = stats.probplot(df[variable], plot=plt)
    plt.show()

"""
VARIABLES FINALES
"""

def preparacion(df):
        
    # Matriz de observaciones
    # X = df.loc[:, "bedrooms":"sqft_lot15"]
    X = df.loc[:, "bedrooms":"distance_to_seattle"]
    
    # Vector etiqueta
    y = df.price
    
    # Array matriz de observaciones
    X = X.iloc[:, 0:].values
    # Variable target
    y = y.values
    
    # Division del conjunto de datos
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    return df, X, y, X_train, X_test, y_train, y_test

"""
AUXILIARES
"""

def function_that_warns():
    warnings.warn("deprecated", DeprecationWarning)

    
def estandar(df):
    scaler = StandardScaler()
    df_std = scaler.fit_transform(df)
    
    df_scaled = pd.DataFrame(df_std, columns=df.columns)
    df = df_scaled
    return df

    
    