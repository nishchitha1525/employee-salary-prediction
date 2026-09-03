from django.shortcuts import render
import pandas as pd
import joblib
import os


# ==========================================
# PROJECT BASE DIRECTORY
# ==========================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)


# ==========================================
# ML MODEL PATH
# ==========================================

MODEL_PATH = os.path.join(
    BASE_DIR,
    "ml_model",
    "salary_model.pkl"
)


# ==========================================
# LOAD ML MODEL
# ==========================================

model = None

if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)


# ==========================================
# SALARY PREDICTION
# ==========================================

def predict_salary(request):

    if request.method == "POST":

        try:

            # Get employee details
            age = int(
                request.POST.get("age")
            )

            gender = request.POST.get(
                "gender"
            )

            education = request.POST.get(
                "education"
            )

            job_title = request.POST.get(
                "job_title"
            )

            experience = int(
                request.POST.get("experience")
            )

            department = request.POST.get(
                "department"
            )


            # Check model
            if model is None:

                return render(
                    request,
                    "predictor/predict.html",
                    {
                        "error":
                        "Machine learning model not found."
                    }
                )


            # Create input data
            input_data = pd.DataFrame({

                "Age": [age],

                "Gender": [gender],

                "Education": [education],

                "Job_Title": [job_title],

                "Experience": [experience],

                "Department": [department]

            })


            # Predict salary
            prediction = model.predict(
                input_data
            )


            # Round prediction
            predicted_salary = round(
                prediction[0],
                2
            )


            # Send employee details
            # and prediction to template
            context = {

                "prediction":
                predicted_salary,

                "form_submitted":
                True,

                "age":
                age,

                "gender":
                gender,

                "education":
                education,

                "job_title":
                job_title,

                "experience":
                experience,

                "department":
                department

            }


            return render(
                request,
                "predictor/predict.html",
                context
            )


        except Exception as e:

            return render(
                request,
                "predictor/predict.html",
                {
                    "error":
                    "Please enter valid information."
                }
            )


    # Display prediction form
    return render(
        request,
        "predictor/predict.html"
    )