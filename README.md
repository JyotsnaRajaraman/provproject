# provproject
Final Project

Data Analysis Provenance & Reproducibility Analysis System

Analyze Jupyter notebooks for reproducibility by executing each cell and comparing outputs with stored results, while checking for missing library installations. For cells containing ML models, assess model bias stability through retraining and explain bias difference above a input threshold using LIME feature importance


Installation 
pip install -r requirements.txt

python nbanalysis.py <notebook_directory>
streamlit run visualizer.py 

#you can upload the analysis json via local web at http://localhost:8501/

