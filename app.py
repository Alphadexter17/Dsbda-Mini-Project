# import streamlit as st
# import numpy as np
# import joblib

# # Load trained components
# model = joblib.load("model.pkl")
# le_exam = joblib.load("le_exam.pkl")
# le_gender = joblib.load("le_gender.pkl")
# le_category = joblib.load("le_category.pkl")
# le_branch = joblib.load("le_branch.pkl")

# # Function to predict with domain logic enforced
# def predict_admission_chances(exam_type, rank, gender, category):
#     try:
#         input_data = [[
#             le_exam.transform([exam_type])[0],
#             rank,
#             le_gender.transform([gender])[0],
#             le_category.transform([category])[0]
#         ]]

#         probabilities = model.predict_proba(input_data)[0]
#         branch_labels = le_branch.inverse_transform(range(len(probabilities)))
#         results_dict = dict(zip(branch_labels, probabilities))

#         # Domain knowledge enforcement: CS > IT > ENTC
#         cs_prob = results_dict.get("CS", 0)
#         it_prob = results_dict.get("IT", 0)
#         entc_prob = results_dict.get("ENTC", 0)

#         if cs_prob > it_prob:
#             results_dict["IT"] = cs_prob
#         if cs_prob > entc_prob:
#             results_dict["ENTC"] = cs_prob

#         results = sorted(results_dict.items(), key=lambda x: x[1], reverse=True)
#         return results

#     except Exception as e:
#         st.error(f"Error: {e}")
#         return []

# # Streamlit UI
# st.title("📊 PICT Admission Chance Predictor")
# st.write("Estimate your chances for CS / IT / ENTC branches based on your exam, rank, gender, and category.")

# exam_type = st.selectbox("Select Exam Type", le_exam.classes_)
# rank = st.number_input("Enter Rank", min_value=1, max_value=300000, value=10000, step=1)
# gender = st.selectbox("Select Gender", le_gender.classes_)
# category = st.selectbox("Select Category", le_category.classes_)

# if st.button("Predict Admission Chances"):
#     results = predict_admission_chances(exam_type, rank, gender, category)

#     if results:
#         st.subheader("📈 Predicted Chances by Branch")
#         i=1
#         for branch, prob in results:
#             st.write(f"**{i}:{branch}**")
#             i=i+1


import streamlit as st
import joblib
import numpy as np
import matplotlib.pyplot as plt

# Load trained model and encoders
model = joblib.load("model.pkl")
le_exam = joblib.load("le_exam.pkl")
le_gender = joblib.load("le_gender.pkl")
le_category = joblib.load("le_category.pkl")
le_branch = joblib.load("le_branch.pkl")

# Title
st.title("🎓 PICT Admission Chance Predictor")

# User Input
exam_type = st.selectbox("Select Entrance Exam Type:", le_exam.classes_)
rank = st.number_input("Enter Your Closing Rank:", min_value=1)
gender = st.selectbox("Select Gender:", le_gender.classes_)
category = st.selectbox("Select Category:", le_category.classes_)

if st.button("Predict Admission Chances"):
    # Encode inputs
    input_data = [[
        le_exam.transform([exam_type])[0],
        rank,
        le_gender.transform([gender])[0],
        le_category.transform([category])[0]
    ]]
    
    # Predict probabilities
    probabilities = model.predict_proba(input_data)[0]
    branch_labels = le_branch.inverse_transform(range(len(probabilities)))

    st.subheader("📊 Admission Chances by Branch:")
    results = list(zip(branch_labels, probabilities))

    # Print probabilities
    for branch, prob in results:
        st.write(f"**{branch}**: {prob:.2%}")
    
    # Bar Chart
    st.subheader("📈 Probability Distribution (Bar Chart)")
    fig, ax = plt.subplots()
    ax.bar(branch_labels, probabilities, color=["#4CAF50", "#2196F3", "#FFC107"])
    ax.set_ylabel("Probability")
    ax.set_ylim([0, 1])
    st.pyplot(fig)

    # Pie Chart (Optional)
    st.subheader("📊 Admission Probability (Pie Chart)")
    fig2, ax2 = plt.subplots()
    ax2.pie(probabilities, labels=branch_labels, autopct="%1.1f%%", startangle=90, colors=["#4CAF50", "#2196F3", "#FFC107"])
    ax2.axis("equal")  # Equal aspect ratio ensures pie is a circle.
    st.pyplot(fig2)
