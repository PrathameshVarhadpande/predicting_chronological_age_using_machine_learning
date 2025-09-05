import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the trained model (replace with your actual model path)
model = joblib.load('best_xgb_model.pkl')  # Replace with your model file path

# Function to get user input and create the feature DataFrame
def get_user_input():
    # Input fields
    st.write("Use formula: (Your Cholesterol Level - 148.8815)/(331.3306-148.8815)")
    cholesterol_level = st.number_input("Cholesterol Level (mg/dL)", min_value=0.0, max_value=500.0, value=200.0, step=0.1)
    st.write("Use formula: (Your BMI - 12.0499)/(43.32987-12.0499)")
    bmi = st.number_input("BMI", min_value=0.0, max_value=50.0, value=25.0, step=0.1)
    st.write("Use formula: (Your Blood Glucose Level - 69.86688)/(185.7361-69.86688)")
    blood_glucose_level = st.number_input("Blood Glucose Level (mg/dL)", min_value=0.0, max_value=500.0, value=100.0, step=0.1)
    st.write("Use formula: (Your Bone Density + 0.21979)/(1.999829+0.21979)")
    bone_density = st.number_input("Bone Density (g/cm²)", min_value=0.0, max_value=5.0, value=1.0, step=0.1)
    st.write("Use formula: (Your Vision Sharpness - 0.2)/(1.062537-0.2)")
    vision_sharpness = st.number_input("Vision Sharpness", min_value=0.0, max_value=10.0, value=8.0, step=0.1)
    st.write("Use formula: (Your Hearing Ability - 0)/(94.00382-0)")
    hearing_ability = st.number_input("Hearing Ability (dB)", min_value=0.0, max_value=100.0, value=20.0, step=0.1)
    st.write("Use formula: (Your Cognitive Function - 30.3821)/(106.4798-30.3821)")
    cognitive_function = st.number_input("Cognitive Function", min_value=0.0, max_value=100.0, value=7.0, step=0.1)
    st.write("Use formula: (Your Stress Level - 1.000428)/(9.996323-1.000428)")
    stress_levels = st.number_input("Stress Levels", min_value=0.0, max_value=10.0, value=5.0, step=0.1)
    st.write("Use formula: (Your Pollution Exposure - 0.006395)/(9.99809-0.006395)")
    pollution_exposure = st.number_input("Pollution Exposure", min_value=0.0, max_value=10.0, value=5.0, step=0.1)
    st.write("Use formula: (Your Sun Exposure - 0.002055)/(11.9925-0.002055)")
    sun_exposure = st.number_input("Sun Exposure", min_value=0.0, max_value=10.0, value=5.0, step=0.1)
    st.write("Use formula: (Your Systolic BP - 97)/(193-97)")
    systolic_bp = st.number_input("Systolic BP", min_value=0.0, max_value=200.0, value=120.0, step=0.1)
    st.write("Use formula: (Your Diastolic BP - 60)/(133-60)")
    diastolic_bp = st.number_input("Diastolic BP", min_value=0.0, max_value=120.0, value=80.0, step=0.1)
    gender_female = st.selectbox("Gender", ["female", "male"], key="gender_female") == "female"
    gender_male = not gender_female

    # Physical Activity Level
    physical_activity = st.selectbox("Physical Activity Level", ["high", "low", "moderate"], key="physical_activity")
    physical_activity_high = physical_activity == "high"
    physical_activity_low = physical_activity == "low"
    physical_activity_moderate = physical_activity == "moderate"

    # Smoking Status
    smoking_status = st.selectbox("Smoking Status", ["current", "former", "never"], key="smoking_status")
    smoking_status_current = smoking_status == "current"
    smoking_status_former = smoking_status == "former"
    smoking_status_never = smoking_status == "never"

    # Alcohol Consumption
    alcohol_consumption = st.selectbox("Alcohol Consumption", ["frequent", "occasional"], key="alcohol_consumption")
    alcohol_consumption_frequent = alcohol_consumption == "frequent"
    alcohol_consumption_occasional = alcohol_consumption == "occasional"

    # Diet
    diet = st.selectbox("Diet", ["balanced", "highfat", "lowcarb", "vegetarian"], key="diet")
    diet_balanced = diet == "balanced"
    diet_highfat = diet == "highfat"
    diet_lowcarb = diet == "lowcarb"
    diet_vegetarian = diet == "vegetarian"

    # Chronic Diseases
    chronic_diseases = st.selectbox("Chronic Diseases", ["diabetes", "heart disease", "hypertension"], key="chronic_diseases")
    chronic_diseases_diabetes = chronic_diseases == "diabetes"
    chronic_diseases_heart_disease = chronic_diseases == "heart disease"
    chronic_diseases_hypertension = chronic_diseases == "hypertension"

    # Medication Use
    medication_use = st.selectbox("Medication Use", ["occasional", "regular"], key="medication_use")
    medication_use_occasional = medication_use == "occasional"
    medication_use_regular = medication_use == "regular"

    # Family History
    family_history = st.selectbox("Family History", ["diabetes", "heart disease", "hypertension"], key="family_history")
    family_history_diabetes = family_history == "diabetes"
    family_history_heart_disease = family_history == "heart disease"
    family_history_hypertension = family_history == "hypertension"

    # Mental Health Status
    mental_health_status = st.selectbox("Mental Health Status", ["excellent", "fair", "good", "poor"], key="mental_health_status")
    mental_health_status_excellent = mental_health_status == "excellent"
    mental_health_status_fair = mental_health_status == "fair"
    mental_health_status_good = mental_health_status == "good"
    mental_health_status_poor = mental_health_status == "poor"

    # Sleep Patterns
    sleep_patterns = st.selectbox("Sleep Patterns", ["excessive", "insomnia", "normal"], key="sleep_patterns")
    sleep_patterns_excessive = sleep_patterns == "excessive"
    sleep_patterns_insomnia = sleep_patterns == "insomnia"
    sleep_patterns_normal = sleep_patterns == "normal"

    # Education Level
    education_level = st.selectbox("Education Level", ["high school", "postgraduate", "undergraduate"], key="education_level")
    education_level_high_school = education_level == "high school"
    education_level_postgraduate = education_level == "postgraduate"
    education_level_undergraduate = education_level == "undergraduate"

    # Income Level
    income_level = st.selectbox("Income Level", ["high", "low", "medium"], key="income_level")
    income_level_high = income_level == "high"
    income_level_low = income_level == "low"
    income_level_medium = income_level == "medium"

    # Age Group
    age_group = st.selectbox("Age Group", ["adult", "middleaged", "senior", "young adult"], key="age_group")
    age_group_adult = age_group == "adult"
    age_group_middleaged = age_group == "middleaged"
    age_group_senior = age_group == "senior"
    age_group_young_adult = age_group == "young adult"

    # BMI Category
    bmi_category = st.selectbox("BMI Category", ["underweight", "normal", "overweight", "obese"], key="bmi_category")
    bmi_category_underweight = bmi_category == "underweight"
    bmi_category_normal = bmi_category == "normal"
    bmi_category_overweight = bmi_category == "overweight"
    bmi_category_obese = bmi_category == "obese"

    # Create the feature DataFrame
    user_input = pd.DataFrame({
        'Cholesterol Level (mg/dL)': [cholesterol_level],
        'BMI': [bmi],
        'Blood Glucose Level (mg/dL)': [blood_glucose_level],
        'Bone Density (g/cm2)': [bone_density],
        'Vision Sharpness': [vision_sharpness],
        'Hearing Ability (dB)': [hearing_ability],
        'Cognitive Function': [cognitive_function],
        'Stress Levels': [stress_levels],
        'Pollution Exposure': [pollution_exposure],
        'Sun Exposure': [sun_exposure],
        'Systolic BP': [systolic_bp],
        'Diastolic BP': [diastolic_bp],
        'Gender_female': [gender_female],
        'Gender_male': [gender_male],
        'Physical Activity Level_high': [physical_activity_high],
        'Physical Activity Level_low': [physical_activity_low],
        'Physical Activity Level_moderate': [physical_activity_moderate],
        'Smoking Status_current': [smoking_status_current],
        'Smoking Status_former': [smoking_status_former],
        'Smoking Status_never': [smoking_status_never],
        'Alcohol Consumption_frequent': [alcohol_consumption_frequent],
        'Alcohol Consumption_occasional': [alcohol_consumption_occasional],
        'Diet_balanced': [diet_balanced],
        'Diet_highfat': [diet_highfat],
        'Diet_lowcarb': [diet_lowcarb],
        'Diet_vegetarian': [diet_vegetarian],
        'Chronic Diseases_diabetes': [chronic_diseases_diabetes],
        'Chronic Diseases_heart disease': [chronic_diseases_heart_disease],
        'Chronic Diseases_hypertension': [chronic_diseases_hypertension],
        'Medication Use_occasional': [medication_use_occasional],
        'Medication Use_regular': [medication_use_regular],
        'Family History_diabetes': [family_history_diabetes],
        'Family History_heart disease': [family_history_heart_disease],
        'Family History_hypertension': [family_history_hypertension],
        'Mental Health Status_excellent': [mental_health_status_excellent],
        'Mental Health Status_fair': [mental_health_status_fair],
        'Mental Health Status_good': [mental_health_status_good],
        'Mental Health Status_poor': [mental_health_status_poor],
        'Sleep Patterns_excessive': [sleep_patterns_excessive],
        'Sleep Patterns_insomnia': [sleep_patterns_insomnia],
        'Sleep Patterns_normal': [sleep_patterns_normal],
        'Education Level_high school': [education_level_high_school],
        'Education Level_postgraduate': [education_level_postgraduate],
        'Education Level_undergraduate': [education_level_undergraduate],
        'Income Level_high': [income_level_high],
        'Income Level_low': [income_level_low],
        'Income Level_medium': [income_level_medium],
        'Age Group_adult': [age_group_adult],
        'Age Group_middleaged': [age_group_middleaged],
        'Age Group_senior': [age_group_senior],
        'Age Group_young adult': [age_group_young_adult],
        'BMI Category_normal': [bmi_category_normal],
        'BMI Category_obese': [bmi_category_obese],
        'BMI Category_overweight': [bmi_category_overweight],
        'BMI Category_underweight': [bmi_category_underweight]
    })
    return user_input

# Display the form to the user
st.title("Chronological Age Prediction Model")
st.write("Team Members: Prathamesh Ravindra Varhadpande, Nikhil Gokhale, Nachiket Shyam Dabhade")
st.write("Please enter your health details:")

# Main code to display Streamlit UI
def main():
    
    # Get user input
    user_input = get_user_input()

    # Create a button to trigger prediction
    if st.button('Predict Age'):
        # If the model is also predicting age
        predicted_age_ = model.predict(user_input)  # Replace with actual age prediction model output if separate
        min_age = 18
        max_age = 89

        # Scale the predicted age back to original age range
        predicted_age = predicted_age_ * (max_age - min_age) + min_age 

        # Rounding the predicted age
        predicted_age_rounded = round(predicted_age[0])


        # Display the predicted age
        st.write(f"Your chronological age based on your health and lifestyle factors is: {predicted_age_rounded}")

if __name__ == "__main__":
    main()