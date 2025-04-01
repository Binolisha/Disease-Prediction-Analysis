import streamlit as st
import numpy as np
import joblib
import os
import sys

# Define paths to model files
model_paths = {
    "Parkinson's Disease": r"C:\Users\DELL\parkinsons_disease_prediction_model.pkl",
    "Liver Disease": r"C:\Users\DELL\liver_disease_prediction_model1.pkl",
    "Kidney Disease": r"C:\Users\DELL\kidney_disease_prediction_model.pkl"
}

# Function to load model
def load_model(disease):
    model_path = model_paths[disease]

    if os.path.exists(model_path):
        try:
            model = joblib.load(model_path)
            return model
        except Exception as e:
            st.error(f"Error loading model: {e}")
            return None
    else:
        st.error(f"Model file not found: {model_path}")
        return None

# Function to get user inputs based on disease selection
def get_user_inputs(disease):
    st.subheader(f"Enter parameters for {disease} prediction")

    if disease == "Parkinson's Disease":
        return np.array([[  
            st.number_input('MDVP:Fo(Hz)', min_value=0.0, value=100.0, step=0.1),
            st.number_input('MDVP:Fhi(Hz)', min_value=0.0, value=150.0, step=0.1),
            st.number_input('MDVP:Flo(Hz)', min_value=0.0, value=120.0, step=0.1),
            st.number_input('MDVP:Jitter(%)', min_value=0.0, value=0.01, step=0.001),
            st.number_input('MDVP:Jitter(Abs)', min_value=0.0, value=0.00001, step=0.00001),
            st.number_input('MDVP:RAP', min_value=0.0, value=0.001, step=0.0001),
            st.number_input('MDVP:PPQ', min_value=0.0, value=0.001, step=0.001),
            st.number_input('Jitter:DDP', min_value=0.0, value=0.001, step=0.001),
            st.number_input('MDVP:Shimmer', min_value=0.0, value=0.03, step=0.1),
            st.number_input('MDVP:Shimmer(dB)', min_value=0.0, value=0.1, step=0.1),
            st.number_input('Shimmer:APQ3', min_value=0.0, value=0.1, step=0.01),
            st.number_input('Shimmer:APQ5', min_value=0.0, value=0.1, step=0.01),
            st.number_input('MDVP:APQ', min_value=0.0, value=0.1, step=0.01),
            st.number_input('Shimmer:DDA', min_value=0.0, value=0.01, step=0.01),
            st.number_input('NHR', min_value=0.0, value=0.001, step=0.001),
            st.number_input('HNR', min_value=0.0, value=20.0, step=1.0),
            st.number_input('RPDE', min_value=0.0, value=0.1, step=0.1),
            st.number_input('DFA', min_value=0.0, value=0.1, step=0.1),
            st.number_input('spread1', min_value=-10.0, value=-1.0, step=1.0),
            st.number_input('spread2', min_value=0.0, value=1.0, step=0.1),
            st.number_input('D2', min_value=0.0, value=1.0, step=0.1),
            st.number_input('PPE', min_value=0.0, value=0.1, step=0.01)
        ]])

    elif disease == "Liver Disease":
        gender_map = {'Male': 0, 'Female': 1}
        gender = gender_map[st.selectbox('Gender', ['Male', 'Female'])]
        return np.array([[  
            st.number_input('Age', min_value=0, value=0, step=1),
            gender,
            st.number_input('Total Bilirubin', min_value=0.0, value=1.0, step=0.1),
            st.number_input('Direct Bilirubin', min_value=0.0, value=0.5, step=0.1),
            st.number_input('Alkaline Phosphotase', min_value=0, value=100, step=1),
            st.number_input('Alamine Aminotransferase', min_value=0, value=50, step=1),
            st.number_input('Aspartate Aminotransferase', min_value=0, value=50, step=1),
            st.number_input('Total Proteins', min_value=0.0, value=6.5, step=0.1),
            st.number_input('Albumin', min_value=0.0, value=3.5, step=0.1),
            st.number_input('Albumin and Globulin Ratio', min_value=0.0, value=1.1, step=0.1)
        ]])

    elif disease == "Kidney Disease":
        Pus_Cells_map={'normal':1,'abnormal':0}
        Pus_cells=Pus_Cells_map[st.selectbox('Pus Cells', ['normal', 'abnormal'])]
        Pus_Cells_Casts_map={'present':1,'notpresent':0}
        Pus_cells_Casts=Pus_Cells_Casts_map[st.selectbox('Pus Cells Casts', ['present', 'notpresent'])]
        Bacteria_map={'present':1,'notpresent':0}
        Bacteria=Bacteria_map[st.selectbox('Bacteria', ['present', 'notpresent'])]
        Hypertension_map={'yes':1,'no':0}
        Hypertension=Hypertension_map[st.selectbox('Hypertension', ['yes', 'no'])]
        Diabetes_Mellitus_map={'yes':1,'no':0}
        Diabetes_Mellitus=Diabetes_Mellitus_map[st.selectbox('Diabetes Mellitus', ['yes', 'no'])]
        Coronary_Artery_Disease_map={'yes':1,'no':0}
        Coronary_Artery_Disease=Coronary_Artery_Disease_map[st.selectbox('Coronary Artery Disease', ['yes', 'no'])]
        Appetite_map={'good':0,'poor':1}
        Appetite=Appetite_map[st.selectbox('Appetite', ['good', 'poor'])]
        Pedal_Edema_map={'yes':1,'no':0}
        Pedal_Edema=Pedal_Edema_map[st.selectbox('Pedal Edema', ['yes', 'no'])]
        Anemia_map={'yes':1,'no':0}
        Anemia=Anemia_map[st.selectbox('Anemia', ['yes', 'no'])]
        return np.array([[  
            st.number_input('Age', min_value=0, value=50, step=1),
            st.number_input('Blood Pressure', min_value=0, value=80, step=10),
            st.number_input('Specific Gravity', min_value=0.0, value=1.02, step=0.01),
            st.number_input('Albumin', min_value=0, value=2, step=1),
            st.number_input('Sugar', min_value=0, value=1, step=1),
            Pus_cells,
            Pus_cells_Casts,
            Bacteria,
            st.number_input('Blood Glucose Random', min_value=0, value=100, step=1),
            st.number_input('Sodium', min_value=100, value=180, step=1),
            st.number_input('Blood Urea', min_value=0, value=40, step=1),
            st.number_input('Serum Creatinine', min_value=0.0, value=1.5, step=0.1),
            st.number_input('Potassium', min_value=0.0, value=1.0, step=0.1),
            st.number_input('Hemoglobin', min_value=0.0, value=12.5, step=0.1),
            st.number_input('Packed Cell Volume', min_value=0, value=40, step=1),
            st.number_input('White Blood Cell Count', min_value=0, value=8000, step=100),
            st.number_input('Red_Blood_Cell_Count', min_value=0.0, value=1.0, step=0.1),
            Hypertension,
            Diabetes_Mellitus,
            Coronary_Artery_Disease,
            Appetite,
            Pedal_Edema,
            Anemia
        ]])

# Main function for Streamlit app
def main():
    st.title("Disease Prediction System")
    
    # Select Disease
    disease = st.selectbox("Select a disease to predict:", list(model_paths.keys()))
    
    # Load the selected model
    model = load_model(disease)

    # Get user input
    user_input = get_user_inputs(disease)

    # Make prediction
    if st.button("Predict"):
        if model:
            try:
                prediction = model.predict(user_input)
                
                # Ensure prediction is an integer
                prediction = int(prediction.item())

                # Display the result
                if prediction == 1:
                    st.success(f"✅ {disease} detected!")
                else:
                    st.success(f"❌ No {disease} detected!")
            except Exception as e:
                st.error(f"⚠️ Prediction failed: {e}")
        else:
            st.error("⚠️ No model available for prediction")

if __name__ == "__main__":
    main()
