import streamlit as st
import cv2
import numpy as np

def process_images():
    cap = cv2.VideoCapture(0)
    azulBajo = np.array([90, 100, 20], np.uint8)
    azulAlto = np.array([120, 255, 255], np.uint8)

    # Configuración de streamlit para video
    video_placeholder = st.empty()
    mensaje_placeholder = st.empty()

    while st.session_state.is_processing:
        ret, frame = cap.read()
        if ret:
            frame = cv2.flip(frame, 1)
            frameHSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            mascara = cv2.inRange(frameHSV, azulBajo, azulAlto)
            contornos, _ = cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(frame, contornos, -1, (255, 0, 0), 4)

            x = 0
            for c in contornos:
                area = cv2.contourArea(c)
                if area > 6000:
                    M = cv2.moments(c)
                    if M["m00"] == 0:
                        M["m00"] = 1
                    x = int(M["m10"] / M["m00"])
                    y = int(M['m01'] / M['m00'])
                    cv2.circle(frame, (x, y), 7, (0, 0, 255), -1)
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    cv2.putText(frame, '{},{}'.format(x, y), (x + 10, y), font, 1.2, (0, 0, 255), 2, cv2.LINE_AA)
                    nuevoContorno = cv2.convexHull(c)
                    cv2.drawContours(frame, [nuevoContorno], 0, (255, 0, 0), 3)

                    # Condiciones para la posición x
                    if x < 100:
                        mensaje = "Mover adelante 100%"
                    elif 200 > x >= 100:
                        mensaje = "Mover adelante 50%"
                    elif 300 > x >= 200:
                        mensaje = "Mover adelante 10%"
                    elif 400 > x >= 300:
                        mensaje = "Mover centro"
                    elif 500 > x >= 400:
                        mensaje = "Mover atrás 30%"
                    elif 600 > x >= 500:
                        mensaje = "Mover atrás 70%"

                    # Actualizar el mensaje centrado en una casilla
                    mensaje_placeholder.markdown(f"<div style='text-align: center; border: 1px solid black; padding: 10px;'>{mensaje}</div>", unsafe_allow_html=True)

            # Convertir de BGR a RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Mostrar el video en la interfaz de Streamlit
            video_placeholder.image(frame_rgb, channels="RGB")

            if cv2.waitKey(1) & 0xFF == ord('s'):
                break

    cap.release()
    cv2.destroyAllWindows()

def main():
    st.title("Camara del robot")

    if st.button("Detener procesamiento"):
        st.session_state.is_processing = False

    # Botón para iniciar y detener el procesamiento
    if st.button("Iniciar procesamiento"):
        st.session_state.is_processing = True
        process_images()

    



if __name__ == "__main__":
    main()
