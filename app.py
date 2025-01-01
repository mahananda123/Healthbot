

from flask import Flask, request, render_template
from flask_sqlalchemy import SQLAlchemy
import pickle
import pandas as pd
import numpy as np
import csv

app = Flask(__name__)

# Configure SQLAlchemy with SQLite database
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///healthcare.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# Database models
class Patient(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    age = db.Column(db.Integer, nullable=False)  # New column
    disease = db.Column(db.String(100), nullable=False)
    description = db.Column(db.Text, nullable=True)
    visiting_link = db.Column(db.String(200), nullable=True)


class Symptom(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    patient_id = db.Column(db.Integer, db.ForeignKey('patient.id'), nullable=False)
    symptom = db.Column(db.String(100), nullable=False)

class Precaution(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    patient_id = db.Column(db.Integer, db.ForeignKey('patient.id'), nullable=False)
    precaution = db.Column(db.String(200), nullable=False)

# Load the trained model and label encoder
with open("svm_model.pkl", "rb") as model_file:
    loaded_model = pickle.load(model_file)

with open("label_encoder.pkl", "rb") as le_file:
    loaded_le = pickle.load(le_file)

# Load additional data for descriptions, precautions, and doctors
def load_csv_to_dict(file_path):
    data_dict = {}
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            if len(row) == 0:  # Skip empty rows
                continue
            key = row[0]
            value = row[1:]
            data_dict[key] = value if len(value) > 1 else value[0] if value else None
    return data_dict

description_dict = load_csv_to_dict('symptom_Description.csv')
precaution_dict = load_csv_to_dict('symptom_precaution.csv')
doctors_dataset = pd.read_csv('doctors_dataset.csv', names=['Name', 'Description'])

# Prepare doctors dataframe
diseases = list(loaded_le.classes_)
doctors = pd.DataFrame({
    "disease": diseases,
    "name": doctors_dataset["Name"],
    "link": doctors_dataset["Description"]
})

# Load training data
training_data = pd.read_csv("Training.csv")
feature_names = training_data.columns[:-1]  # Exclude the target column
grouped_symptoms = training_data.groupby('prognosis').max()

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/questions", methods=["POST"])
def questions():
    name = request.form["name"]
    age = int(request.form["age"])  # Get the age
    disease_or_symptom = request.form["input"].strip().lower()

    grouped_symptoms.index = grouped_symptoms.index.str.lower()
    symptom_list = feature_names.str.lower().tolist()

    symptoms = []
    if disease_or_symptom in grouped_symptoms.index:
        relevant_symptoms = grouped_symptoms.loc[disease_or_symptom]
        symptoms = [symptom for symptom, present in relevant_symptoms.items() if present > 0]
    else:
        symptoms = symptom_list

    return render_template(
        "questions.html", name=name, age=age, disease_or_symptom=disease_or_symptom, symptoms=symptoms
    )
@app.route("/predict", methods=["POST"])
def predict():
    name = request.form["name"]
    age = request.form["age"]  # Get the age
    symptoms = request.form.getlist("symptoms")
    input_vector = np.zeros(len(feature_names))

    for symptom in symptoms:
        if symptom in feature_names:
            symptom_index = list(feature_names).index(symptom)
            input_vector[symptom_index] = 1

    prediction = loaded_model.predict([input_vector])
    predicted_disease = loaded_le.inverse_transform(prediction)[0]

    description = description_dict.get(predicted_disease, "No description available.")
    precautions = precaution_dict.get(predicted_disease, [])
    doctor_row = doctors[doctors['disease'] == predicted_disease]

    patient = Patient(
        name=name, age=age, disease=predicted_disease, description=description,
        visiting_link=doctor_row['link'].values[0]
    )
    db.session.add(patient)
    db.session.commit()

    for symptom in symptoms:
        db.session.add(Symptom(patient_id=patient.id, symptom=symptom))
    for precaution in precautions:
        db.session.add(Precaution(patient_id=patient.id, precaution=precaution))
    
    db.session.commit()

    return render_template(
        "result.html", name=name, age=age, disease=predicted_disease,
        description=description, precautions=precautions,
        doctor_name=doctor_row['name'].values[0], doctor_link=doctor_row['link'].values[0]
    )

@app.route("/view-data")
def view_data():
    # Fetch all patients and related symptoms & precautions
    patients = Patient.query.all()
    patient_data = []
    for patient in patients:
        symptoms = Symptom.query.filter_by(patient_id=patient.id).all()
        precautions = Precaution.query.filter_by(patient_id=patient.id).all()
        patient_data.append({
            "name": patient.name,
            "age":patient.age,
            "disease": patient.disease,
            "description": patient.description,
            "visiting_link": patient.visiting_link,
            "symptoms": [s.symptom for s in symptoms],
            "precautions": [p.precaution for p in precautions]
        })
    
    return render_template("view_data.html", patient_data=patient_data)


if __name__ == "__main__":
    with app.app_context():
        db.create_all()  # Create tables if they don't exist
    app.run(debug=True)
