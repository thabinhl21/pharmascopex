import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import plotly.express as px
import gdown

st.title('Precision Oncology')
st.write('##### Find out which drugs are most effective against a chosen cell line using the dropdown menu below')


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


df = pd.read_csv('../CellLine_DrugName.csv')
# model_ic50 = joblib.load("dd_rf_ic50.pkl")
# model_auc = joblib.load("dd_rf_auc.pkl")


# prepare dropdown options
cell_mapping = df[['CELL_LINE_NAME', 'COSMIC_ID']].dropna().drop_duplicates()
cell_lines = sorted(cell_mapping['CELL_LINE_NAME'].unique())
selected_cell_line = st.selectbox("Select a Cell Line", cell_lines)

# match selected cell line to cosmic_id
selected_cosmic_id = cell_mapping.loc[cell_mapping['CELL_LINE_NAME'] == selected_cell_line, 'COSMIC_ID'].values[0]

# filter dataset for selected cell line
df_filtered = df[df['COSMIC_ID'] == selected_cosmic_id].copy()

# predict if there are rows available
if df_filtered.empty:
    st.warning("No data found for this cell line.")
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
        hover_name='DRUG_NAME', 
        title=f"{selected_cell_line} — Predicted Cell Line Sensitivity Across Drugs",
        labels={
            'Pred_ln_IC50': 'Predicted ln(IC50)',
            'Pred_AUC': 'Predicted AUC'
        }
    )

    fig.update_traces(marker=dict(size=8, opacity=0.7))
    st.plotly_chart(fig)

