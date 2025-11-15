import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import plotly.express as px
import gdown

st.title('Drug Discovery')
st.write('##### Find out which cell lines are most sensitive to a chosen drug using the dropdown menu below')

# download pkl files (models) from google drive
@st.cache_resource
def load_ic50_model():
    file_id = '1_UnZ3zwfmIwRzVYXgnj7V0UzOy9MtR6g'
    url = f"https://drive.google.com/uc?id={file_id}"
    output = 'rf_ic50.pkl'
    gdown.download(url, output, quiet=False)
    return joblib.load(output)


@st.cache_resource
def load_auc_model():
    file_id = '1HoNXAT_7qumWQ6uLMiQZ2Tw9LeZRUjTR'
    url = f"https://drive.google.com/uc?id={file_id}"
    output = 'rf_auc.pkl'
    gdown.download(url, output, quiet=False)
    return joblib.load(output)

model_ic50 = load_ic50_model()
model_auc = load_auc_model()

df = pd.read_csv('visualization/CellLine_DrugName.csv')
# model_ic50 = joblib.load("dd_rf_ic50.pkl")
# model_auc = joblib.load("dd_rf_auc.pkl")


# prepare dropdown options
drug_mapping = df[['DRUG_NAME', 'DRUG_ID']].dropna().drop_duplicates()
drug_names = sorted(drug_mapping['DRUG_NAME'].unique())
selected_drug_name = st.selectbox("Select a Drug", drug_names)

# match selected drug name to drug_id
selected_drug_id = drug_mapping.loc[drug_mapping['DRUG_NAME'] == selected_drug_name, 'DRUG_ID'].values[0]

# filter dataset for selected drug
df_filtered = df[df['DRUG_ID'] == selected_drug_id].copy()

# predict if there are rows available
if df_filtered.empty:
    st.warning("No data found for this drug.")
else:
    # features for prediction
    X = df_filtered[['DRUG_ID', 'COSMIC_ID']]

    # make predictions
    df_filtered['Pred_ln_IC50'] = model_ic50.predict(X)
    df_filtered['Pred_AUC'] = model_auc.predict(X)

    # plot interactive scatterplot with cell line hover
    fig = px.scatter(
        df_filtered,
        x='Pred_ln_IC50',
        y='Pred_AUC',
        hover_name='CELL_LINE_NAME', 
        title=f"{selected_drug_name} — Predicted Drug Response Across Cell Lines",
        labels={
            'Pred_ln_IC50': 'Predicted ln(IC50)',
            'Pred_AUC': 'Predicted AUC'
        }
    )

    fig.update_traces(marker=dict(size=8, opacity=0.7))
    st.plotly_chart(fig)

