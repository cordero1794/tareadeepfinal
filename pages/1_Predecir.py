import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow import keras
import streamlit as st
import requests

st.title("Predecir los valores de los Grados del robot y la Dirección.")

# URL del archivo CSV en GitHub
csv_url = 'https://raw.githubusercontent.com/cordero1794/MODELO-DH/main/coordenadas%20xyz2.csv'

# Descargar el archivo CSV desde GitHub
response = requests.get(csv_url)



# Guardar el contenido del archivo CSV descargado
with open('coordenadas_xyz2.csv', 'wb') as f:
    f.write(response.content)

# Leer los datos del archivo CSV
data = pd.read_csv('coordenadas_xyz2.csv')


# Obtener las coordenadas del usuario con st.form
with st.form(key="coord_form"):
    st.subheader("Introduce las coordenadas para hacer la predicción:")
    input_x = st.slider("Valor de X", min_value=-1.0, max_value=1.0, value=0.0, step=0.01)
    input_y = st.slider("Valor de Y", min_value=-1.0, max_value=1.0, value=0.0, step=0.01)
    input_z = st.slider("Valor de Z", min_value=-1.0, max_value=1.0, value=0.0, step=0.01)
    submit_button = st.form_submit_button(label="Realizar Predicción")

# Procesar los datos después de hacer clic en el botón de envío
if submit_button:
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

    # Agregar las nuevas coordenadas introducidas por el usuario
    nuevas_coordenadas = np.array([[input_x, input_y, input_z]])
    st.write("Nuevas coordenadas:", nuevas_coordenadas)

    # Cargar el modelo de dirección entrenado
    model_direccion = keras.models.load_model('./saved_models/modeloRedNeuronal_Direccion2.h5')

    # Cargar el modelo de ángulos entrenado
    model_theta = keras.models.load_model('./saved_models/modeloRedNeuronal_Theta2.h5')

    # Predicción de dirección
    prediction_direccion = model_direccion.predict(scaler.transform(nuevas_coordenadas))
    predicted_direction = label_encoder_direccion.inverse_transform([np.argmax(prediction_direccion)])
    st.write(f"Dirección predicha: {predicted_direction[0]}")

    # Predicción de ángulos
    prediction_theta = model_theta.predict(scaler.transform(nuevas_coordenadas))
    st.write(f"Valores de Theta1, Theta2, Theta3, Theta4 predichos: {prediction_theta[0]}")

   
