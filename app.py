from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

app = Flask(__name__)

MODEL_PATH = Path(__file__).resolve().parent / "modelo_superconductividad.pkl"

paquete_modelo = joblib.load(MODEL_PATH)

modelo = paquete_modelo["pipeline"]
feature_names = paquete_modelo["feature_names"]
defaults = paquete_modelo["defaults"]
target = paquete_modelo.get("target", "critical_temp")
target_unit = paquete_modelo.get("target_unit", "K")


# Nombres descriptivos para las propiedades del dataset
nombres_propiedades = {
    "atomic_mass": "masa atómica",
    "fie": "energía de primera ionización",
    "atomic_radius": "radio atómico",
    "Density": "densidad",
    "ElectronAffinity": "afinidad electrónica",
    "FusionHeat": "calor de fusión",
    "ThermalConductivity": "conductividad térmica",
    "Valence": "valencia"
}

# Nombres descriptivos para los cálculos estadísticos
nombres_estadisticos = {
    "wtd_gmean": "Media geométrica ponderada",
    "wtd_entropy": "Entropía ponderada",
    "wtd_range": "Rango ponderado",
    "wtd_std": "Desviación estándar ponderada",
    "wtd_mean": "Media ponderada",
    "gmean": "Media geométrica",
    "entropy": "Entropía",
    "range": "Rango",
    "std": "Desviación estándar",
    "mean": "Media"
}


def crear_label_descriptivo(nombre_columna):
    """
    Convierte nombres técnicos del dataset en nombres más comprensibles
    para mostrarlos en el formulario.
    """

    if nombre_columna == "number_of_elements":
        return "Número de elementos químicos del material"

    for prefijo, descripcion_prefijo in nombres_estadisticos.items():
        patron = prefijo + "_"

        if nombre_columna.startswith(patron):
            propiedad = nombre_columna.replace(patron, "", 1)
            descripcion_propiedad = nombres_propiedades.get(
                propiedad,
                propiedad.replace("_", " ")
            )

            return f"{descripcion_prefijo} de {descripcion_propiedad}"

    return nombre_columna.replace("_", " ")


def crear_descripcion_campo(nombre_columna):
    """
    Crea una descripción breve para cada campo del formulario.
    """

    if nombre_columna == "number_of_elements":
        return "Cantidad de elementos químicos que componen el material superconductor."

    return f"Variable original del dataset: {nombre_columna}. Ingresa un valor numérico real del material."


field_labels = {
    col: crear_label_descriptivo(col)
    for col in feature_names
}

field_descriptions = {
    col: crear_descripcion_campo(col)
    for col in feature_names
}


@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    errors = []

    # Valores iniciales del formulario: medianas del dataset
    values = {
        col: ''
        for col in feature_names
    }

    if request.method == "POST":
        row = {}

        for col in feature_names:
            raw_value = request.form.get(col, "").strip()
            values[col] = raw_value

            # Si el usuario deja un campo vacío, el pipeline lo imputará
            if raw_value == "":
                row[col] = np.nan
                continue

            try:
                row[col] = float(raw_value)
            except ValueError:
                errors.append(
                    f"El campo '{field_labels.get(col, col)}' debe contener un valor numérico válido."
                )

        if not errors:
            try:
                input_df = pd.DataFrame([row], columns=feature_names)
                pred = modelo.predict(input_df)[0]
                prediction = round(float(pred), 4)
            except Exception as e:
                errors.append(f"Ocurrió un error al generar la predicción: {str(e)}")

    return render_template(
        "index.html",
        feature_names=feature_names,
        field_labels=field_labels,
        field_descriptions=field_descriptions,
        values=values,
        prediction=prediction,
        errors=errors,
        target=target,
        target_unit=target_unit
    )


@app.route("/health")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    app.run(port= 8001, debug=True)