from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Cargar el pipeline y las columnas originales
modelo = joblib.load("modelo_ridge_pca.joblib")
columnas_entrada = joblib.load("columnas_entrada.joblib")


@app.route("/", methods=["GET", "POST"])
def index():
    prediccion = None
    error = None

    if request.method == "POST":
        try:
            datos = {
                "age": int(request.form["age"]),
                "gender": request.form["gender"],
                "study_hours_per_day": float(request.form["study_hours_per_day"]),
                "social_media_hours": float(request.form["social_media_hours"]),
                "netflix_hours": float(request.form["netflix_hours"]),
                "part_time_job": request.form["part_time_job"],
                "attendance_percentage": float(request.form["attendance_percentage"]),
                "sleep_hours": float(request.form["sleep_hours"]),
                "diet_quality": request.form["diet_quality"],
                "exercise_frequency": int(request.form["exercise_frequency"]),
                "parental_education_level": request.form["parental_education_level"],
                "internet_quality": request.form["internet_quality"],
                "mental_health_rating": int(request.form["mental_health_rating"]),
                "extracurricular_participation": request.form["extracurricular_participation"]
            }

            # Validaciones
            if not 0 <= datos["age"] <= 100:
                raise ValueError("La edad debe estar entre 0 y 100.")

            if not 0 <= datos["study_hours_per_day"] <= 24:
                raise ValueError("Las horas de estudio deben estar entre 0 y 24.")

            if not 0 <= datos["social_media_hours"] <= 24:
                raise ValueError("Las horas en redes sociales deben estar entre 0 y 24.")

            if not 0 <= datos["netflix_hours"] <= 24:
                raise ValueError("Las horas de Netflix deben estar entre 0 y 24.")

            if not 0 <= datos["attendance_percentage"] <= 100:
                raise ValueError("La asistencia debe estar entre 0 y 100.")

            if not 0 <= datos["sleep_hours"] <= 24:
                raise ValueError("Las horas de sueño deben estar entre 0 y 24.")

            if not 0 <= datos["exercise_frequency"] <= 7:
                raise ValueError("La frecuencia de ejercicio debe estar entre 0 y 7.")

            if not 1 <= datos["mental_health_rating"] <= 10:
                raise ValueError("La salud mental debe estar entre 1 y 10.")

            # Crear DataFrame con el mismo orden de columnas usado en entrenamiento
            entrada = pd.DataFrame([datos])
            entrada = entrada[columnas_entrada]

            # Predicción
            resultado = modelo.predict(entrada)[0]

            # Limitar resultado a rango lógico de calificación
            resultado = max(0, min(100, resultado))

            prediccion = round(resultado, 2)

        except Exception as e:
            error = str(e)

    return render_template("index.html", prediccion=prediccion, error=error)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8000, debug=True)