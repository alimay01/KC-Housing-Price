import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import folium
from folium.plugins import HeatMap

from plotnine import ggplot, aes, geom_point, coord_flip, labs, ggtitle, theme_bw

import warnings


"""
FUNCIONES PRINCIPALES
"""

def explore_data(filepath):
    """
    Función que muestra información básica acerca de un dataframe.
    
    Parameters:
    -----------
    df: Pandas DataFrame
        El dataframe que se desea explorar.
    """
    # Cargamos los datos
    df = load_data(filepath)
    
    # Mostrar las primeras filas del dataframe
    print("Mostramos todas las columnas de las primeras 10 filas del dataframe:")
    pd.options.display.max_columns = None # para que se nos muestren todas las columnas
    display(df.head(10))
    print("-"*50)
    
    # Mostrar el tamaño del dataframe
    print("Tamaño del dataframe:")
    print("Número de filas:", df.shape[0])
    print("Número de columnas:", df.shape[1])
    print("-"*50)
    
    # Mostrar información sobre las columnas del dataframe
    print("Información de las columnas:")
    display(df.info())
    print("-"*50)
    
    # Ver si hay valores nulos
    print("Valores nulos:")
    display(df.isnull().sum())
    print("-"*50)
    
    return df

def describe_data(df):
    """
    Funcón que realiza un descriptivo del DataFrame, junto a algunos gráficos.
    """
    
    # Descriptivo del dataframe
    print("Mostramos una tabla con los principales estadísticos:")
    pd.options.display.float_format = '{:.4f}'.format # fijamos un número de decimales
    display(df.describe())      # así evitamos notación científica (ej. 2.161300e+04) que no se interpreta rápido
    print("-"*50)
    
    # Histogramas del dataframe
    print("Y realizamos unos histogramas sobre todo el conjunto de datos:")
    df.hist(bins=50, figsize=(20,15))
    display(df.plot())
    print("-"*50)
   
    
def graphic_data(df):
    """
    Función principal que muestra algunas visualizaciones exploratorias
    """
    # Scatter plot 'price' frente a 'view'
    print("Price en función de view:")
    scatter_plot(df, 'view', 'price')
    print("-"*50)
    
    # Scatter plot 'price' frente a 'view'
    print("Price en función de bathrooms:")
    scatter_plot(df, 'bathrooms', 'price')
    print("-"*50)
    
    # Segmentacion por 'waterfront'
    print("Segmentacion por waterfront:")
    segmentar(df, "waterfront")
    
    barplot(df, "waterfront", "price")
    print("-"*50)
    
    # Segmentacion por 'grade'
    print("Segmentacion por grade:")
    segmentar(df, "grade")

    barplot(df, "grade", "price")
    print("-"*50)
    
    
    
"""
FUNCIONES AUXILIARES
"""

def function_that_warns():
    warnings.warn("deprecated", DeprecationWarning)

def load_data(file_path):
    """
    Función que carga los datos desde un archivo CSV y devuelve un DataFrame de Pandas
    """
    df = pd.read_csv(file_path)
    return df

def plot_histogram(df, variable):
    """
    Función que crea un histograma para una variable dada del DataFrame dado
    """
    plt.hist(df[variable])
    plt.xlabel(variable)
    plt.ylabel('Frecuencia')
    plt.title('Histograma de ' + variable)

def plot_boxplot(df, variable_x, variable_y):
    """
    Función que crea un boxplot para dos variables dadas del DataFrame dado
    """
    sns.boxplot(x=variable_x, y=variable_y, data=df)
    plt.title("Boxplot de " +  variable_y + " segmentando por " + variable_x)
    plt.show()
    
def niveles(df):
    valores_distintos = pd.DataFrame(columns=['variable', 'valores_distintos'])

    for col in df.columns:
        valores = df[col].unique()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            valores_distintos = valores_distintos.append({'variable': col, 'valores_distintos': valores}, ignore_index=True)
    
    print(valores_distintos.to_string(index=False))
    
def scatter_plot(df, variable_x, variable_y):
    p = ggplot(df, aes(x=variable_x, y=variable_y, color=variable_x)) + \
        geom_point(size=1, alpha=0.5) + \
        coord_flip() + \
        labs(x=variable_x, y=variable_y, color=variable_x) + \
        ggtitle('House prices by ' + variable_y) + \
        theme_bw()
    display(p)
    
    
def segmentar(df, variable):
    # Segmentacion
    data = df.groupby(by=[variable])
    
    # Descriptivo
    data.describe()
    
def barplot(df, variable_x, variable_y):
    
    # Generar un gráfico de barras y utilizar diferentes colores para cada valor único de waterfront
    sns.barplot(data=df, x=variable_x, y=variable_y, palette="crest")
    
    # Configurar el título y las etiquetas de los ejes
    plt.title(variable_y + ' by ' + variable_x)
    plt.xlabel(variable_x)
    plt.ylabel(variable_y)
    
    # Quitar los bordes del gráfico
    sns.despine()
    
    # Mostrar el gráfico
    plt.show()


    
