import scipy
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt 

import xgboost as xgb
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_squared_error
from sklearn.inspection import permutation_importance

"""
XGBOOSTER
"""

def XGB(df, X,y,X_train, y_train, X_test, y_test):
    data_matrix = xgb.DMatrix(data=X, label=y)
    xg_reg = xgb.XGBRegressor(colsample_bytree=0.8, learning_rate=0.1, max_depth=6, n_estimators=1000, verbosity=0)
    xg_reg.fit(X_train, y_train)
    y_pred = xg_reg.predict(X_test)
    xgb_mse, xgb_rmse, xgb_r2, xgb_mae = metricas(y_test, y_pred)
    xg_reg.get_booster().feature_names = df.iloc[:, 1:].columns.to_list()
    xgb.plot_importance(xg_reg)
    return xgb_mse, xgb_rmse, xgb_r2, xgb_mae
    
    
"""
REGRESION LINEAL
"""

def reg_lin(X_train, y_train, X_test, y_test):
    # Crear el modelo de regresión lineal
    model = LinearRegression()
    model.fit(X_train, y_train)    
    y_pred = model.predict(X_test)
    
    lin_mse, lin_rmse, lin_r2, lin_mae = metricas(y_test, y_pred)    
    return lin_mse, lin_rmse, lin_r2, lin_mae
    
"""
KNN
"""

def KNN_h(X_train, y_train, X_test, y_test):
    # Creamos el modelo y hiperparametrizamos
    knn = KNeighborsRegressor()
    
    param_grid = {
        'n_neighbors': [3, 5, 7, 9, 11]         # número de vecinos
    }
    
    grid_search = GridSearchCV(knn, param_grid=param_grid, cv=5)
    grid_search.fit(X_train, y_train)
    
    print("Mejores parámetros: ", grid_search.best_params_)
    print("Mejor puntuación de validación cruzada: {:.2f}".format(grid_search.best_score_))
    
    # Hacer predicciones en los datos de prueba
    y_pred = grid_search.predict(X_test)
    
    knn_mse, knn_rmse, knn_r2, knn_mae = metricas(y_test, y_pred)    
    return knn_mse, knn_rmse, knn_r2, knn_mae
    
"""
ARBOL DE DECISION
"""

def arbol_decision_h(X_train, y_train, X_test, y_test):
    # Crear un modelo de árbol de decisión
    tree = DecisionTreeRegressor()
    # Hiperparametrizar y entrenar el modelo con los datos de entrenamiento
    param_grid = {
        'max_depth': [None, 5, 10, 15, 20],
        'min_samples_split': [2, 5, 10, 15, 20],
        'min_samples_leaf': [1, 2, 4, 8, 16]
    }
    
    grid_search = GridSearchCV(tree, param_grid=param_grid, cv=5)
    grid_search.fit(X_train, y_train)
    
    print("Mejores parámetros: ", grid_search.best_params_)
    print("Mejor puntuación de validación cruzada: {:.2f}".format(grid_search.best_score_))
    
    # Hacer predicciones en los datos de prueba
    y_pred = grid_search.predict(X_test)
    
    dt_mse, dt_rmse, dt_r2, dt_mae = metricas(y_test, y_pred)
    return dt_mse, dt_rmse, dt_r2, dt_mae
    
    
"""
BAGGING
"""

def bagging_h(X_train, y_train, X_test, y_test):
    # Crear el modelo de árbol de decisión
    tree = DecisionTreeRegressor()
    
    bagging = BaggingRegressor(base_estimator=tree, random_state=42)
    
    params = {
        'n_estimators': [5, 10, 15],
        'base_estimator__max_depth': [5, 10, 15],
        'base_estimator__min_samples_split': [2, 5, 10],
        'base_estimator__min_samples_leaf': [1, 2, 4]
    }
    
    grid_search = GridSearchCV(estimator=bagging, param_grid=params, cv=5)
    grid_search.fit(X_train, y_train)
    
    print("Mejores parámetros: ", grid_search.best_params_)
    print("Mejor puntuación de validación cruzada: {:.2f}".format(grid_search.best_score_))
    
    y_pred = grid_search.predict(X_test)
    
    bagging_mse, bagging_rmse, bagging_r2, bagging_mae = metricas(y_test, y_pred)
    return bagging_mse, bagging_rmse, bagging_r2, bagging_mae


"""
RANDOM FOREST
"""

def random_forest(X_train, y_train, X_test, y_test):
    # Crear el modelo de random forest
    rf_reg = RandomForestRegressor(random_state=42)
    
    # Realizar una hiperparametrización
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'max_features': [2, 4, 6]
    }
    
    grid_search = GridSearchCV(rf_reg, param_grid=param_grid, cv=5, n_jobs=-1)
    
    # Ajustar el modelo a los datos de entrenamiento
    grid_search.fit(X_train, y_train)
    
    print("Mejores parámetros: ", grid_search.best_params_)
    print("Mejor puntuación de validación cruzada: {:.2f}".format(grid_search.best_score_))
    
    # Hacer predicciones en los datos de prueba 
    y_pred = grid_search.predict(X_test)

    rf_mse, rf_rmse, rf_r2, rf_mae = metricas(y_test, y_pred)
    return rf_mse, rf_rmse, rf_r2, rf_mae


"""
GRADIENT BOOST
"""

def gradient_boost(X_train, y_train, X_test, y_test):

    # modelo de regresión con Gradient Boosting
    gb_reg = GradientBoostingRegressor(random_state=42)
    
    # definir los valores de los hiperparámetros a explorar
    param_grid = {                       
        'n_estimators': [100, 200, 300],                            # numero de arboles
        'max_depth': [3, 4, 5],                                     # profundidad maxima de cada arbol
        'learning_rate': [0.1, 0.05, 0.01]                          # tamaño de paso 
    }
    
    # instanciar el objeto GridSearchCV
    grid_search = GridSearchCV(estimator=gb_reg, param_grid=param_grid, cv=5, n_jobs=-1)
    
    # ajustar el objeto GridSearchCV a los datos de entrenamiento
    grid_search.fit(X_train, y_train)
    
    # obtener el mejor modelo
    best_gb_reg = grid_search.best_estimator_
    
    # hacer predicciones en los datos de prueba
    y_pred = best_gb_reg.predict(X_test)
    
    gb_mse, gb_rmse, gb_r2, gb_mae = metricas(y_test, y_pred)
    return gb_mse, gb_rmse, gb_r2, gb_mae    

def orden_metricas(metrics_table):
    """
    Funcion que nos realiza ordenaciones según las distintas columnas
    """
    # MSE
    print("MSE de menor a mayor :")
    ordenar(metrics_table, 'MSE')
    print("-"*50)
    
    #MAE
    print("MAE de menor a mayor :")
    ordenar(metrics_table, 'MAE')
    print("-"*50)
    
    #RMSE
    print("RMSE de menor a mayor :")
    ordenar(metrics_table, 'RMSE')
    print("-"*50)
    
    #R2
    print("R2 de menor a mayor :")
    ordenar(metrics_table, 'R2')
    print("-"*50)

    
"""
AUXILIARES
"""

def metricas(y_test, y_pred):
    # Calcular predicciones y errores
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    
    # Imprimir los resultados
    print("MSE: {:.2f}".format(mse))         # Nos interesa que sea pequeño
    print("RMSE: {:.2f}".format(rmse))       # Nos interesa que sea pequeño
    print("R2: {:.2f}".format(r2))           # Nos interesa que sea grande
    print("MAE: {:.2f}".format(mae))         # Nos interesa que sea pequeño
    
    return mse, rmse, r2, mae

def ordenar(metrics_table, columna):
    
    # Ordenar la tabla por la columna "MSE" de menor a mayor
    metrics_table_sorted = metrics_table.sort_values(by=columna)
    
    # Mostrar la tabla ordenada
    display(metrics_table_sorted)
    
    
    
    
    