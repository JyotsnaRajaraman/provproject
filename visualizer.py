import streamlit as st
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys

def load_results(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def plot_reproducibility_summary(results):
    """Plot reproducibility summary for all notebooks."""
    total_reproducible = sum(n['summary']['reproducible_cells'] for n in results.values())
    total_random = sum(n['summary']['reproducible_random_cells'] for n in results.values())
    total_non_reproducible = sum(n['summary']['non_reproducible_cells'] for n in results.values())
    total_failed_random = sum(n['summary']['non_reproducible_random_cells'] for n in results.values())

    fig = go.Figure(data=[go.Pie(
        labels=['Reproducible', 'Reproducible Random', 'Non-reproducible', 'Failed Random'],
        values=[total_reproducible, total_random, total_non_reproducible, total_failed_random],
        hole=.3,
        marker_colors=['#2ecc71', '#3498db', '#e74c3c', '#f39c12']
    )])
    return fig

def plot_non_reproducible_breakdown(non_reproducible_details):
    """Plot breakdown of non-reproducible cells by category."""
    categories = {
        'Execution Errors': len(non_reproducible_details['execution_errors']),
        'Output Mismatches': len(non_reproducible_details['output_mismatches']),
        'Missing Outputs': len(non_reproducible_details['missing_expected_outputs']),
        'Unexpected Outputs': len(non_reproducible_details['unexpected_new_outputs']),
        'Failed Random': len(non_reproducible_details['failed_random_cells']),
        'Model Bias Variations': len(non_reproducible_details.get('model_bias_variations', []))  # Added this line
    }

    fig = go.Figure(data=[go.Bar(
        x=list(categories.keys()),
        y=list(categories.values()),
        marker_color='#e74c3c'
    )])

    fig.update_layout(
        title='Non-reproducible Cells by Category',
        xaxis_title='Category',
        yaxis_title='Number of Cells'
    )
    return fig

def plot_model_analysis(models):
    """Plot bias comparison for models."""
    return None

def display_non_reproducible_cells(notebook_results):
    """Display detailed information about non-reproducible cells."""
    st.subheader("Non-reproducible Cells Details")

    tabs = st.tabs([
        "Execution Errors",
        "Output Mismatches",
        "Missing Outputs",
        "Unexpected Outputs",
        "Failed Random Cells",
        "Model Bias Variations"  # Added new tab
    ])

    details = notebook_results.get('non_reproducible_details', {})

    with tabs[0]:
        if details.get('execution_errors'):
            for error in details['execution_errors']:
                st.error(f"Cell {error['cell_number']}: {error['error']}")
        else:
            st.info("No execution errors found")

    with tabs[1]:
        if details.get('output_mismatches'):
            for mismatch in details['output_mismatches']:
                st.markdown(f"**Cell {mismatch['cell_number']}**")
                col1, col2 = st.columns(2)
                with col1:
                    st.write("Original Output:")
                    st.json(mismatch['original_output'])
                with col2:
                    st.write("New Output:")
                    st.json(mismatch['new_output'])
                st.divider()
        else:
            st.info("No output mismatches found")

    with tabs[2]:
        if details.get('missing_expected_outputs'):
            for missing in details['missing_expected_outputs']:
                st.markdown(f"**Cell {missing['cell_number']}**")
                st.write("Expected Output:")
                st.json(missing['expected_output'])
                st.divider()
        else:
            st.info("No missing outputs found")

    with tabs[3]:
        if details.get('unexpected_new_outputs'):
            for unexpected in details['unexpected_new_outputs']:
                st.markdown(f"**Cell {unexpected['cell_number']}**")
                st.write("Unexpected Output:")
                st.json(unexpected['unexpected_output'])
                st.divider()
        else:
            st.info("No unexpected outputs found")

    with tabs[4]:
        if details.get('failed_random_cells'):
            for failed in details['failed_random_cells']:
                st.error(f"Cell {failed['cell_number']}: {failed['failure_details']['reason']}")
        else:
            st.info("No failed random cells found")

    with tabs[5]:  # New tab for model bias variations
        if details.get('model_bias_variations'):
            for variation in details['model_bias_variations']:
                st.markdown(f"**Cell {variation['cell_number']}**")
                bias_details = variation['bias_details']
                
                # Create a comparison table
                bias_df = pd.DataFrame({
                    'Metric': ['Original Bias', 'Retrained Bias', 'Bias Difference', 'Threshold'],
                    'Value': [
                        f"{bias_details['original_bias']:.3f}",
                        f"{bias_details['retrained_bias']:.3f}",
                        f"{bias_details['bias_difference']:.3f}",
                        f"{bias_details['threshold']:.3f}"
                    ]
                })
                st.table(bias_df)
                
                # Add a visual indicator
                if bias_details['bias_difference'] > bias_details['threshold']:
                    st.warning(f"⚠️ Bias difference ({bias_details['bias_difference']:.3f}) exceeds threshold ({bias_details['threshold']:.3f})")
                
                st.divider()
        else:
            st.info("No model bias variations found")

def display_model_analysis(models):
    """Display detailed model analysis information."""
    if not models:
        st.info("No ML models found in notebook")
        return

    st.subheader("Model Analysis Details")

    for model_id, details in models.items():
        with st.expander(f"Model Details"):
            # LIME Analysis moved up and displayed side by side
            if details.get('lime_analysis'):
                st.subheader("LIME Analysis")
                col1, col2 = st.columns(2)

                with col1:
                    if details['lime_analysis'].get('original_explanation'):
                        st.write("Feature Importance (Original Model):")
                        st.table(pd.DataFrame(details['lime_analysis']['original_explanation']))

                with col2:
                    if details['lime_analysis'].get('retrained_explanation'):
                        st.write("Feature Importance (Retrained Model):")
                        st.table(pd.DataFrame(details['lime_analysis']['retrained_explanation']))

            # Basic model information below LIME analysis
            st.write("Model Type:", details.get('model_type', 'Unknown'))
            st.write("Cell Number:", details.get('cell_number', 'Unknown'))
            if details.get('parameters'):
                st.write("Model Parameters:")
                st.json(details['parameters'])

def main():
    st.set_page_config(page_title="Notebook Provenance Analysis", layout="wide")
    st.title("Notebook Provenance Analysis Dashboard")

    uploaded_file = st.file_uploader("Choose a results file", type='json')
    if uploaded_file:
        results = json.load(uploaded_file)

        st.header("Overall Summary")
        col1, col2 = st.columns(2)

        with col1:
            st.metric("Total Notebooks", len(results))
            total_cells = sum(n['summary']['total_cells'] for n in results.values())
            st.metric("Total Cells", total_cells)
            total_non_reproducible = sum(n['summary']['non_reproducible_cells'] for n in results.values())
            st.metric("Non-reproducible Cells", total_non_reproducible)

        with col2:
            fig = plot_reproducibility_summary(results)
            st.plotly_chart(fig)

        st.header("Notebook Details")
        for notebook_path, notebook_results in results.items():
            st.markdown(f"### Notebook: {notebook_path}")
            with st.container():
                col1, col2 = st.columns(2)

                with col1:
                    st.subheader("Library Analysis")
                    st.write("Missing Libraries:", notebook_results['library_analysis']['missing'])

                    if 'non_reproducible_details' in notebook_results:
                        fig = plot_non_reproducible_breakdown(notebook_results['non_reproducible_details'])
                        st.plotly_chart(fig)

                with col2:
                    st.subheader("Cell Statistics")
                    st.write(notebook_results['summary'])

                if notebook_results.get('non_reproducible_details'):
                    display_non_reproducible_cells(notebook_results)

                # Updated ML analysis section
                if notebook_results['ml_analysis'].get('models'):
                    st.subheader("Machine Learning Analysis")
                    bias_fig = plot_model_analysis(notebook_results['ml_analysis']['models'])
                    if bias_fig:
                        st.plotly_chart(bias_fig)
                    display_model_analysis(notebook_results['ml_analysis']['models'])

                st.divider()

if __name__ == "__main__":
    main()
