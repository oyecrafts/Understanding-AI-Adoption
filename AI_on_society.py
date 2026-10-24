# -*- coding: utf-8 -*-
"""
Created on Sat Oct 24 15:31:10 2026

@author: Admin
"""
import pandas as pd
import csv
AI=pd.read_csv(r"C:\Users\Admin\OneDrive\Documents\Python\AI_on_society.csv",encoding="cp1252")

#read the dataset
print(AI.head())
Column_names=AI.columns

# perform data cleaning
null_values=AI.isnull().sum()
print(null_values)
#60 null values were returned under column 
#What is your occupation? (optional).This was accepted and analysis was continued
print(AI.dtypes)
#all data types were correctly formatted to string
#id column was dropped from analysis
AI_cleaned=AI.drop(columns="ID")

#“Ön Bachelor's degree” appears to be a mixture of Turkish and English.
print(AI_cleaned["What is your education level?"].value_counts(dropna=False))
AI_cleaned["What is your education level?"] = AI_cleaned["What is your education level?"].replace({
    "Ön Bachelor's degree": "Associate degree"
})
#renamed columns that will be used for analysis
Columns_to_keep= AI_cleaned.rename(
    columns={
        "What is your age range?": "age_group",
        "What is your gender?": "gender",
        "What is your education level?": "education_level",
        "What is your employment status?": "employment_status",
        "What is your occupation? (optional)": "occupation",
        "How much knowledge do you have about artificial intelligence (AI) technologies?": "ai_knowledge",
        "Do you generally trust artificial intelligence (AI)?": "ai_trust",
        "Please rate how actively you use AI-powered products in your daily life on a scale from 1 to 5.": "ai_usage",
        "Would you like to use more AI products in the future?": "future_willingness","Do you think your own job could be affected by artificial intelligence (AI)?": "perceived_job_impact"})
desired_columns = [
    "age_group",
    "gender",
    "education_level",
    "employment_status",
    "occupation",
    "ai_knowledge",
    "ai_trust",
    "ai_usage",
    "future_willingness",
    "perceived_job_impact",
]
AI_Analysis = Columns_to_keep[desired_columns]

print(AI_Analysis["ai_knowledge"].unique())

knowledge_mapping = {
    "I have no knowledge": 1,
    "I've heard a little about it": 2,
    "I have basic knowledge": 3,
    "I have a good level of knowledge": 4,
    "I have an expert-level knowledge": 5
}
AI_Analysis["knowledge_score"] = (
    AI_Analysis["ai_knowledge"].map(knowledge_mapping)
)

print(AI_Analysis["ai_trust"].unique())
trust_mapping = {
    "I don't trust it at all": 1,
    "I don't trust it": 2,
    "I'm undecided": 3,
    "I trust it": 4,
    "I fully trust it": 5
}
AI_Analysis["trust_score"] = (
    AI_Analysis["ai_trust"].map(trust_mapping)
)

print(AI_Analysis["future_willingness"].unique())
willingness_mapping = {
    "Definitely, I would not like to": 1,
    "I would not like to": 2,
    "I'm undecided": 3,
    "I would like to": 4,
    "Definitely, I would like to": 5
}
AI_Analysis["willingness_score"] = (
    AI_Analysis["future_willingness"].map(willingness_mapping)
)
print(AI_Analysis["perceived_job_impact"].unique())

job_impact_mapping = {
    "Definitely I don't think so": 1,
    "I don't think so": 2,
    "I'm undecided": 3,
    "Think": 4,
    "I definitely think": 5}

AI_Analysis["job_impact_score"] = (
    AI_Analysis["perceived_job_impact"].map(job_impact_mapping)
)

print(
    AI_Analysis[
        ["perceived_job_impact", "job_impact_score"]
    ].head(10)
)

print(AI_Analysis["job_impact_score"].isna().sum())

def frequency_table(data, column):
    counts = data[column].value_counts(dropna=False)
    percentages = (
        data[column]
        .value_counts(normalize=True, dropna=False)
        .mul(100)
        .round(1)
    )

    return pd.DataFrame({
        "Frequency": counts,
        "Percentage": percentages
    })

print(frequency_table(AI_Analysis, "age_group"))
print(frequency_table(AI_Analysis, "ai_knowledge"))
print(frequency_table(AI_Analysis, "ai_trust"))
print(frequency_table(AI_Analysis, "ai_usage"))
print(frequency_table(AI_Analysis, "future_willingness"))

import streamlit as st
import pandas as pd

# 1. Save the cleaned dataframe FIRST (before Streamlit runs)
AI_Analysis.to_csv(r"C:\Users\Admin\OneDrive\Documents\Python\AI_Analysis.csv", index=False)
print("Data successfully cleaned and saved!")

# 2. Now start your Streamlit code
import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="Understanding AI Adoption",
    layout="wide"
)

st.title("Understanding AI Adoption")
st.write(
    "Exploring the relationship between demographics, "
    "AI knowledge, trust and willingness to adopt AI."
)

# 3. Read using the absolute path
df = pd.read_csv(r"C:\Users\Admin\OneDrive\Documents\Python\AI_Analysis.csv", encoding="cp1252")

# Display a preview in Streamlit
st.dataframe(df.head())





