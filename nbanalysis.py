import nbformat
import json
import datetime
import ast
import importlib
import pkg_resources
import re
from sklearn.datasets import load_iris
from nbconvert.preprocessors import ExecutePreprocessor
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split 
import numpy as np
import pandas as pd
from collections import defaultdict
from lime import lime_tabular
import os
from typing import Dict, List, Optional, Tuple
from firstpassanalysis import NotebookAnalyzer


class NotebookProvenanceAnalyzer:
    def __init__(self, notebook_dir: str, bias_threshold: float = 0.1):
        self.notebook_dir = notebook_dir
        self.bias_threshold = bias_threshold
        self.execution_processor = ExecutePreprocessor(timeout=600, kernel_name='python3')
        
    def find_notebooks(self) -> list[str]:
        """Find all Jupyter notebooks in the directory."""
        print(f'inside find_notebooks, self.notebook_dir: {self.notebook_dir}')
        nbs = []
        for f in os.listdir(self.notebook_dir):
            if f.endswith('.ipynb'):
                nbs.append(os.path.join(self.notebook_dir, f))
        return nbs

    def analyze_notebooks(self) -> Dict:
        """Analyze all notebooks in the directory."""
        results = {}
        notebooks = self.find_notebooks()
        for notebook_path in notebooks:
            analyzer = NotebookAnalyzer(notebook_path, self.bias_threshold)
            results[notebook_path] = self._enhance_analysis(analyzer)
        return results

    def _enhance_analysis(self, analyzer):
        """Enhance the existing analysis with additional features."""
        results = analyzer.analyze_notebook()

        # Add LIME analysis for ML models
        if results['ml_analysis']['models']:
            for model_id, details in results['ml_analysis']['models'].items():
                # Skip if LIME analysis already exists
                if not details.get('lime_analysis'):
                    lime_results = self._perform_lime_analysis(details, analyzer.notebook)
                    if lime_results:
                        details['lime_analysis'] = lime_results
        return results

    def _perform_lime_analysis(self, model_details, notebook):
        """Perform LIME analysis on original and retrained models."""
        try:
            cell_idx = model_details['cell_number']
            
            # Initialize namespace with required imports
            ns = {
                'np': np,
                'pd': pd,
                'RandomForestClassifier': RandomForestClassifier,
                'load_iris': load_iris,
                'train_test_split': train_test_split,
                'accuracy_score': accuracy_score
            }
            
            # Execute all cells up to and including the model cell
            executed_cells = []
            for idx, cell in enumerate(notebook.cells[:cell_idx + 1]):
                if cell.cell_type == 'code' and not cell.source.startswith('!pip'):
                    try:
                        # Skip pip install cells
                        if not cell.source.strip().startswith('!pip'):
                            exec(cell.source, ns)
                            executed_cells.append(cell.source)
                    except Exception as e:
                        print(f"Error executing cell {idx}: {str(e)}")
                        continue

            # Verify we have all required components
            model = ns.get(model_details['variable_name'])
            X_train = ns.get('X_train')
            X_test = ns.get('X_test')
            y_train = ns.get('y_train')

            if not all([model, X_train is not None, X_test is not None, y_train is not None]):
                print(f"Missing required data for LIME analysis in cell {cell_idx}")
                return None

            # Create LIME explainer with proper data formatting
            training_data = X_train.values if hasattr(X_train, 'values') else X_train
            feature_names = X_train.columns.tolist() if hasattr(X_train, 'columns') else None
            
            explainer = lime_tabular.LimeTabularExplainer(
                training_data=training_data,
                feature_names=feature_names,
                mode='classification',
                training_labels=y_train,
                verbose=False
            )

            # Get first instance for explanation
            sample_instance = X_test.iloc[0].values if hasattr(X_test, 'iloc') else X_test[0]

            # Generate explanations for original model
            orig_exp = explainer.explain_instance(
                sample_instance,
                model.predict_proba,
                num_features=len(feature_names) if feature_names else 10,
                num_samples=100
            )

            # Retrain model with same parameters
            new_model = model.__class__(**model.get_params())
            new_model.fit(X_train, y_train)
            
            # Generate explanations for retrained model
            new_exp = explainer.explain_instance(
                sample_instance,
                new_model.predict_proba,
                num_features=len(feature_names) if feature_names else 10,
                num_samples=100
            )

            # Format and return results
            lime_results = {
                'original_explanation': orig_exp.as_list(),
                'retrained_explanation': new_exp.as_list(),
                'feature_importance_diff': self._compare_explanations(
                    orig_exp.as_list(),
                    new_exp.as_list()
                ),
                'prediction_probabilities': {
                    'original': model.predict_proba(sample_instance.reshape(1, -1))[0].tolist(),
                    'retrained': new_model.predict_proba(sample_instance.reshape(1, -1))[0].tolist()
                }
            }

            return lime_results

        except Exception as e:
            print(f"Error in LIME analysis for cell {cell_idx}: {str(e)}")
            import traceback
            print(traceback.format_exc())  # Print full traceback for debugging
            return None

    def _compare_explanations(self, orig_exp, new_exp):
        """Compare LIME explanations between original and retrained models."""
        try:
            orig_dict = dict(orig_exp)
            new_dict = dict(new_exp)

            all_features = set(orig_dict.keys()) | set(new_dict.keys())
            differences = {}

            for feature in all_features:
                orig_val = orig_dict.get(feature, 0)
                new_val = new_dict.get(feature, 0)
                abs_diff = abs(orig_val - new_val)
                
                if abs_diff > 0.01:  # Threshold for significant difference
                    differences[feature] = {
                        'original_importance': float(orig_val),  # Convert to float for JSON serialization
                        'retrained_importance': float(new_val),
                        'absolute_difference': float(abs_diff)
                    }

            return differences

        except Exception as e:
            print(f"Error in comparing explanations: {str(e)}")
            return {}

    def _compare_outputs(self, original_outputs, new_outputs):
        """Compare cell outputs considering different output types."""
        if len(original_outputs) != len(new_outputs):
            return False

        for orig, new in zip(original_outputs, new_outputs):
            if orig['output_type'] != new['output_type']:
                return False

            if orig['output_type'] == 'execute_result':
                if orig['data'] != new['data']:
                    return False
            elif orig['output_type'] == 'stream':
                if orig['text'] != new['text']:
                    return False
            elif orig['output_type'] == 'display_data':
                if orig['data'] != new['data']:
                    return False

        return True

    def generate_report(self, results, output_path=None):
        """Generate a comprehensive report of the analysis."""
        report = {
            'timestamp': datetime.datetime.now().isoformat(),
            'summary': {
                'total_notebooks': len(results),
                'total_cells': sum(r['summary']['total_cells'] for r in results.values()),
                'reproducible_cells': sum(r['summary']['reproducible_cells'] for r in results.values()),
                'models_analyzed': sum(len(r['ml_analysis']['models']) for r in results.values())
            },
            'notebook_results': results
        }

        if output_path:
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2)

        return report


def analyze_directory(directory_path: str, output_path: Optional[str] = None, 
                     bias_threshold: float = 0.1) -> Dict:
    if os.path.isfile(directory_path) and directory_path.endswith('.ipynb'):
        analyzer = NotebookAnalyzer(directory_path, bias_threshold)
        results = {directory_path: analyzer.analyze_notebook()}
    else:
        analyzer = NotebookProvenanceAnalyzer(directory_path, bias_threshold)
        results = analyzer.analyze_notebooks()
    
    if output_path:
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
    
    return results

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python nbanalysis.py <notebook_directory>")
        sys.exit(1)
        
    directory_path = sys.argv[1]
    output_path = os.path.join(f"./analysis_results_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    
    analyzer = NotebookProvenanceAnalyzer(directory_path, bias_threshold=0.05)
    results = analyzer.analyze_notebooks()
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Analysis complete. Results saved to {output_path}")
