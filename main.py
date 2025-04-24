# import pandas as pd
# import numpy as np

# # For reproducibility
# np.random.seed(42)

# # Number of records
# n = 250

# # Randomly generate fields
# exam_types = np.random.choice(["JEE", "MHT-CET"], n, p=[0.4, 0.6])
# percentiles = np.round(np.random.uniform(60, 100, n), 2)
# categories = np.random.choice(["Open", "OBC", "SC", "ST"], n, p=[0.5, 0.25, 0.15, 0.10])
# home_universities = np.random.choice(["Pune", "Other"], n, p=[0.7, 0.3])

# # Admission logic with category-wise cutoffs
# def get_admission_status(exam, percentile, category):
#     cutoffs = {
#         "Open":    {"MHT-CET": 96, "JEE": 98},
#         "OBC":     {"MHT-CET": 94, "JEE": 96},
#         "SC":      {"MHT-CET": 92, "JEE": 94},
#         "ST":      {"MHT-CET": 90, "JEE": 92}
#     }
#     return 1 if percentile >= cutoffs[category][exam] else 0

# # Generate admission status
# admission_status = [
#     get_admission_status(exam_types[i], percentiles[i], categories[i])
#     for i in range(n)
# ]

# # Create DataFrame
# df = pd.DataFrame({
#     "Exam": exam_types,
#     "Percentile": percentiles,
#     "Category": categories,
#     "Home_University": home_universities,
#     "Admission_Status": admission_status
# })

# # Display first few records
# print(df.head())

# # Optionally, save to CSV
# df.to_csv("pict_admission_dataset.csv", index=False)

# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.preprocessing import LabelEncoder
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import classification_report, confusion_matrix

# np.random.seed(42)

# # Step 1: Generate dataset
# n = 250
# exam_types = np.random.choice(["JEE", "MHT-CET"], n, p=[0.4, 0.6])
# percentiles = np.round(np.random.uniform(60, 100, n), 2)
# categories = np.random.choice(["Open", "OBC", "SC", "ST"], n, p=[0.5, 0.25, 0.15, 0.10])
# home_universities = np.random.choice(["Pune", "Other"], n, p=[0.7, 0.3])

# def get_admission_status(exam, percentile, category):
#     cutoffs = {
#         "Open": {"MHT-CET": 96, "JEE": 98},
#         "OBC":  {"MHT-CET": 94, "JEE": 96},
#         "SC":   {"MHT-CET": 92, "JEE": 94},
#         "ST":   {"MHT-CET": 90, "JEE": 92}
#     }
#     return 1 if percentile >= cutoffs[category][exam] else 0

# admission_status = [
#     get_admission_status(exam_types[i], percentiles[i], categories[i])
#     for i in range(n)
# ]

# df = pd.DataFrame({
#     "Exam": exam_types,
#     "Percentile": percentiles,
#     "Category": categories,
#     "Home_University": home_universities,
#     "Admission_Status": admission_status
# })

# # Step 2: Data cleaning
# df.drop_duplicates(inplace=True)
# df.dropna(inplace=True)

# # Step 3: Label encoding
# df['Exam'] = LabelEncoder().fit_transform(df['Exam'])
# df['Category'] = LabelEncoder().fit_transform(df['Category'])
# df['Home_University'] = LabelEncoder().fit_transform(df['Home_University'])

# # Step 4: Visualizations
# sns.histplot(df['Percentile'], kde=True, bins=20)
# plt.title("Distribution of Percentile")
# plt.show()

# sns.countplot(x='Category', hue='Admission_Status', data=df)
# plt.title("Admission Status by Category")
# plt.show()

# sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
# plt.title("Correlation Heatmap")
# plt.show()

# # Step 5: Model training
# X = df.drop('Admission_Status', axis=1)
# y = df['Admission_Status']
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# model = LogisticRegression()
# model.fit(X_train, y_train)

# # Step 6: Evaluation
# y_pred = model.predict(X_test)
# print("Classification Report:\n", classification_report(y_test, y_pred))
# print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))


import numpy as np
import pandas as pd
# Set seed for reproducibility
np.random.seed(42)

# Constants for synthetic realistic generation
num_samples = 500
exam_types = ["MHT-CET", "JEE"]
genders = ["Male", "Female"]
categories = ["Open", "OBC", "SC", "ST"]
branches = ["CS", "IT", "ENTC"]

# Targeted distributions
category_dist = [0.5, 0.27, 0.15, 0.08]  # Open, OBC, SC, ST
branch_dist = [0.4, 0.35, 0.25]         # CS > IT > ENTC
gender_dist = [0.55, 0.45]              # Slightly more males
exam_type_dist = [0.65, 0.35]           # More from MHT-CET than JEE

# Helper to generate realistic closing ranks
def generate_closing_rank(exam, category, branch):
    base = {
        "MHT-CET": {
            "Open": {"CS": 1200, "IT": 2500, "ENTC": 4000},
            "OBC": {"CS": 3000, "IT": 4500, "ENTC": 6000},
            "SC": {"CS": 7000, "IT": 8500, "ENTC": 10000},
            "ST": {"CS": 8000, "IT": 9500, "ENTC": 11000},
        },
        "JEE": {
            "Open": {"CS": 2000, "IT": 3500, "ENTC": 5000},
            "OBC": {"CS": 4000, "IT": 5500, "ENTC": 7000},
            "SC": {"CS": 8000, "IT": 9500, "ENTC": 11000},
            "ST": {"CS": 9000, "IT": 10500, "ENTC": 12000},
        }
    }

    rank_mean = base[exam][category][branch]
    return int(np.clip(np.random.normal(rank_mean, 600), 100, 15000))

# Generate the dataset
realistic_data = {
    "Exam_Type": np.random.choice(exam_types, size=num_samples, p=exam_type_dist),
    "Gender": np.random.choice(genders, size=num_samples, p=gender_dist),
    "Category": np.random.choice(categories, size=num_samples, p=category_dist),
    "Branch": np.random.choice(branches, size=num_samples, p=branch_dist),
}

# Compute ranks using the function
realistic_data["Closing_Rank"] = [
    generate_closing_rank(exam, cat, br)
    for exam, cat, br in zip(realistic_data["Exam_Type"], realistic_data["Category"], realistic_data["Branch"])
]

# Create DataFrame and save
df_realistic = pd.DataFrame(realistic_data)
output_path = "realistic_admission_data.csv"
df_realistic.to_csv(output_path, index=False)

df_realistic.head()
