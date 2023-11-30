import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow import keras
from tensorflow.keras import layers
import streamlit as st
import os
import requests
from tqdm import tqdm

st.set_page_config(page_title="Modelo de DH", page_icon="🤖")
st.title("Modelo para predecir el efector final del robot.Implementando el Algoritmo de Denavit-Hartenberg")


# URL del archivo CSV en GitHub
csv_url = 'https://raw.githubusercontent.com/cordero1794/MODELO-DH/main/coordenadas%20xyz2.csv'

# Botón para Iniciar Entrenamiento
if st.button("Iniciar Entrenamiento"):
    # Descargar el archivo CSV desde GitHub
    response = requests.get(csv_url)

    # Imprimir el estado de la descarga
    st.write(f"Estado de la descarga: {response.status_code}")

    # Guardar el contenido del archivo CSV descargado
    with open('coordenadas_xyz2.csv', 'wb') as f:
        f.write(response.content)

    # Leer los datos del archivo CSV
    data = pd.read_csv('coordenadas_xyz2.csv')

    # Imprimir las primeras filas del DataFrame
    st.write("Primeras filas del DataFrame:")
    st.dataframe(data.head())

    # Preprocesar los datos
    X = data[['X', 'Y', 'Z']]
    y_direccion = data['Direccion']
    y_theta = data[['Theta1', 'Theta2', 'Theta3', 'Theta4']]

    # Codificar las etiquetas de dirección (convertir a valores numéricos)
    label_encoder_direccion = LabelEncoder()
    y_encoded_direccion = label_encoder_direccion.fit_transform(y_direccion)

    # Escalar las características (opcional, pero puede mejorar el rendimiento)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Dividir los datos
    X_train, X_test, y_train_direccion, y_test_direccion, y_train_theta, y_test_theta = train_test_split(
        X_scaled, y_encoded_direccion, y_theta, test_size=0.2, random_state=42
    )

    # Definir el modelo de dirección
    model_direccion = keras.Sequential([
        layers.Input(shape=(3,)),
        layers.Dense(64, activation='relu'),
        layers.Dense(4, activation='softmax', name='direccion')
    ])

    # Compilar el modelo de dirección
    model_direccion.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Entrenar el modelo de dirección
    
    progress_direccion = st.progress(0)
    st.text("Entrenando el modelo de dirección...")
    for epoch in range(300):
        history_direccion = model_direccion.fit(X_train, y_train_direccion, epochs=1, batch_size=32, validation_split=0.2, verbose=0)
        progress_direccion.progress((epoch + 1) / 300)

    st.text("Entrenamiento del modelo de dirección completado.")

    # Evaluar el modelo de dirección
    accuracy_direccion = model_direccion.evaluate(X_test, y_test_direccion)
    st.write(f"Precisión de la dirección: {accuracy_direccion[1] * 100:.2f}%")

    # Definir un modelo para la regresión de ángulos
    model_theta = keras.Sequential([
        layers.Input(shape=(3,)),
        layers.Dense(64, activation='relu'),
        layers.Dense(4, activation='linear', name='theta')
    ])

    # Compilar el modelo para la regresión de ángulos
    model_theta.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

    # Entrenar el modelo para la regresión de ángulos
    progress_theta = st.progress(0)
    st.text("Entrenando el modelo para la regresión de ángulos...")
    for epoch in range(300):
        history_theta = model_theta.fit(X_train, y_train_theta, epochs=1, batch_size=32, validation_split=0.2, verbose=0)
        progress_theta.progress((epoch + 1) / 300)

    st.text("Entrenamiento del modelo para la regresión de ángulos completado.")

    # Evaluar el modelo de ángulos
    accuracy_theta = model_theta.evaluate(X_test, y_test_theta)
    st.write(f"Error absoluto medio de ángulos: {accuracy_theta[1]:.2f}")

    # Guardar los modelos en una carpeta
    ruta_carpeta_guardado = './saved_models'
    os.makedirs(ruta_carpeta_guardado, exist_ok=True)

    nombre_archivo_modelo_direccion = 'modeloRedNeuronal_Direccion2.h5'
    nombre_archivo_modelo_theta = 'modeloRedNeuronal_Theta2.h5'
    
    # Guardar modelo de dirección
    model_direccion.save(os.path.join(ruta_carpeta_guardado, nombre_archivo_modelo_direccion))
    st.write(f"Modelo de Dirección guardado en: {os.path.join(ruta_carpeta_guardado, nombre_archivo_modelo_direccion)}")

    # Guardar modelo de ángulos
    model_theta.save(os.path.join(ruta_carpeta_guardado, nombre_archivo_modelo_theta))
    st.write(f"Modelo de Theta guardado en: {os.path.join(ruta_carpeta_guardado, nombre_archivo_modelo_theta)}")





