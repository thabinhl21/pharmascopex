import streamlit as st
import pandas as pd
import joblib
# import onnxruntime as ort
import plotly.express as px
import gdown
from ParetoFront_AnalysisPlotly import plot_pareto_front_analysis

st.set_page_config(layout="wide")

# to center align title and header
st.markdown(
    """
    <h1 style='text-align: center; font-size: 42px;'>Drug Discovery</h1>
    <p style='text-align: center; font-size: 18px;'>Find out which cell lines are most sensitive to a chosen drug using the dropdown menu below</p>
    """,
    unsafe_allow_html=True
)

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

@st.cache_resource
def load_sens_model():
    file_id = '1BD151EccNg1ks_sPjov1C7qEQQR1hlqg'
    url = f"https://drive.google.com/uc?id={file_id}"
    output = 'rf_sens.pkl'
    gdown.download(url, output, quiet=False)
    return joblib.load(output)

model_ic50 = load_ic50_model()
model_auc = load_auc_model()
model_sens = load_sens_model()

df = pd.read_csv('visualization/CellLine_DrugName.csv')

# prepare dropdown options
drug_mapping = df[['DRUG_NAME', 'DRUG_ID']].dropna().drop_duplicates()
drug_names = sorted(drug_mapping['DRUG_NAME'].unique())
selected_drug_name = st.selectbox("Select a Drug: type to search or scroll to browse", drug_names)

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
    df_filtered['Sensitivity'] = model_sens.predict(X)

    # plot interactive scatterplot with cell line hover
    fig = px.scatter(
        df_filtered,
        x='Pred_ln_IC50',
        y='Pred_AUC',
        hover_name='CELL_LINE_NAME', 
        render_mode='svg',
        # title=f"{selected_drug_name} — Predicted Drug Response Across Cell Lines",
        labels={
            'Pred_ln_IC50': 'Predicted ln(IC50)',
            'Pred_AUC': 'Predicted AUC'
        }
    )


    # compute Pareto front
    pareto_fig, pareto_x, pareto_y, pareto_idx = plot_pareto_front_analysis(
        df_filtered['Pred_ln_IC50'],
        df_filtered['Pred_AUC'],
        show_plot=False
    )

    # add Pareto front line
    fig.add_scatter(
        x=pareto_x,
        y=pareto_y,
        mode="lines",
        line=dict(color="red", width=2),
        name="Pareto Front"
    )

    # add Pareto front points
    fig.add_scatter(
        x=pareto_x,
        y=pareto_y,
        mode="markers",
        marker=dict(size=12, color="orangered", opacity=0.7, line=dict(color='darkred', width=1.5)),
        name="Pareto Optimal Points"
    )

    fig.update_traces(marker=dict(size=8, opacity=0.7))

    # to display 2 visualizations side by sidee
    col1, col2 = st.columns([.6, .4])

    # left visualization: scatter plot + pareto front
    with col1:
        # st.markdown(
        #     f"<h3 style='text-align:center;'>{selected_drug_name} — Predicted Drug Response Across Cell Lines</h3>",
        #     unsafe_allow_html=True
        # )

        fig.update_layout(
            margin=dict(t=80),
            title={
                'text':f"{selected_drug_name} - Predicted Drug Response Across Cell Lines",
                'x': 0.5,
                'xanchor': 'center'
            }
        )

        st.plotly_chart(fig, use_container_width=True)

    # right visualization: bar chart of top 5 cell lines
    cell_scores = (
        df_filtered
        .groupby('CELL_LINE_NAME', as_index=False)['Sensitivity']
        .min()
    )

    top5 = cell_scores.nsmallest(5, 'Sensitivity')
    # st.text(top5.to_string())

    bar_fig = px.bar(
        top5,
        x='Sensitivity',
        y='CELL_LINE_NAME',
        orientation='h',
        labels={'Sensitivity': 'Sensitivity Score', 'CELL_LINE_NAME': ''},
    )

    # bar_fig.update_traces(textposition='inside', insidetextanchor='middle')
    bar_fig.update_traces(
        texttemplate='%{x:.3f}',
        textposition='inside',
        insidetextanchor='middle',
        hovertemplate='<b>%{y}</b><br>Sensitivity: %{x:.4f}<extra></extra>'
    )

    bar_fig.update_layout(
        height=400,
        width=700,
        margin=dict(t=80),
        yaxis=dict(autorange="reversed"),  
        title={
            'text':f"Top 5 Most Sensitive Cell Lines",
            'x': 0.5,
            'xanchor': 'center'
        }
    )

    with col2:
        # st.markdown(
        #     "<h3 style='text-align:center;'>Top 5 Most Sensitive Cell Lines</h3>",
        #     unsafe_allow_html=True
        # )
        st.plotly_chart(bar_fig, use_container_width=True)
        st.markdown(
            """
                <p style='font-size:12px; text-align:left;'>
                Sensitivity Score: weighted average of ln(IC50) & AUC -> (0.5*ln(IC50) + 0.5*AUC).
                </p>
            """
            """
                <p style='font-size:12px; text-align:left;'>
                Lower (more negative) values = higher predicted sensitivity to the drug
                </p>
            """,
            unsafe_allow_html=True
        )

