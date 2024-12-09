import nbformat
import json
import datetime
import ast
import importlib
import pkg_resources
import re
from nbconvert.preprocessors import ExecutePreprocessor
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split  # Added this import
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import numpy as np
import pandas as pd
from collections import defaultdict
from lime import lime_tabular
import os
from typing import Dict, List, Optional, Tuple

class NotebookAnalyzer:
    def __init__(self, notebook_path, bias_threshold=0.1):
        self.notebook_path = notebook_path
        self.bias_threshold = bias_threshold
        self.results = {
            'metadata': {
                'notebook_path': notebook_path,
                'timestamp': datetime.datetime.now().isoformat(),
                'bias_threshold': bias_threshold
            },
            'summary': {
                'total_cells': 0,
                'reproducible_cells': 0,
                'non_reproducible_cells': 0,
                'random_cells': 0,
                'reproducible_random_cells': 0,
                'non_reproducible_random_cells': 0,
                'no_output_cells': 0
            },
            'non_reproducible_details': {
                'execution_errors': [],
                'output_mismatches': [],
                'missing_expected_outputs': [],
                'unexpected_new_outputs': [],
                'failed_random_cells': [],
                'model_bias_variations': []
            },
            'library_analysis': {
                'installed': [],
                'required': [],
                'missing': []
            },
            'cell_results': [],
            'ml_analysis': {
                'models': {}
            }
        }

        self.base_namespace = {
            'np': np,
            'pd': pd,
            'RandomForestClassifier': RandomForestClassifier,
            'GradientBoostingClassifier': GradientBoostingClassifier,
            'LogisticRegression': LogisticRegression,
            'SVC': SVC,
            'train_test_split': train_test_split,
            'accuracy_score': accuracy_score
        }

        with open(notebook_path, 'r') as f:
            self.notebook = nbformat.read(f, as_version=4)

        self.original_outputs = self._analyze_original_outputs()

    def _record_non_reproducible_cell(self, cell_number, reason, details):
        """Record detailed information about a non-reproducible cell."""
        if reason == 'execution_error':
            self.results['non_reproducible_details']['execution_errors'].append({
                'cell_number': cell_number,
                'error': details
            })
        elif reason == 'output_mismatch':
            self.results['non_reproducible_details']['output_mismatches'].append({
                'cell_number': cell_number,
                'original_output': details['original'],
                'new_output': details['new']
            })
        elif reason == 'missing_output':
            self.results['non_reproducible_details']['missing_expected_outputs'].append({
                'cell_number': cell_number,
                'expected_output': details
            })
        elif reason == 'unexpected_output':
            self.results['non_reproducible_details']['unexpected_new_outputs'].append({
                'cell_number': cell_number,
                'unexpected_output': details
            })
        elif reason == 'failed_random':
            self.results['non_reproducible_details']['failed_random_cells'].append({
                'cell_number': cell_number,
                'failure_details': details
            })
        elif reason == 'model_bias_variation':
            self.results['non_reproducible_details']['model_bias_variations'].append({
                'cell_number': cell_number,
                'bias_details': details
            })

    def is_import_only_cell(self, cell_source):
        """Check if a cell contains only import statements and comments."""
        import ast
        try:
            # Parse the cell source into an AST
            tree = ast.parse(cell_source)

            # Check top-level body nodes only
            for node in tree.body:
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    continue
                elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Str):
                    # This catches docstrings and other string expressions
                    continue
                else:
                    # Found a non-import, non-comment statement
                    return False
            return True
        except Exception as e:
            # If parsing fails, assume it's not an import-only cell
            return False

    def is_pip_install_only_cell(self, cell_source):
        """Check if a cell contains only '!pip install ...' statements."""
        try:
            # Split the cell source into lines
            lines = cell_source.strip().splitlines()

            # Check each line to see if it starts with '!pip install'
            for line in lines:
                stripped_line = line.strip()
                # Allow empty lines or lines that start with '#'
                if not stripped_line or stripped_line.startswith('#'):
                    continue
                # Check if the line starts with '!pip install'
                if not stripped_line.startswith('!pip install'):
                    return False
            return True
        except Exception as e:
            # If there's an unexpected issue, assume it's not a pip install only cell
            return False



    def execute_and_compare_cells(self):
        """Execute notebook and compare with original outputs."""
        ep = ExecutePreprocessor(timeout=600, kernel_name='python3')
        exec_nb = nbformat.v4.new_notebook(cells=self.notebook.cells)

        try:
            ep.preprocess(exec_nb)
            for idx, (orig_cell, exec_cell) in enumerate(zip(self.notebook.cells, exec_nb.cells)):
                if orig_cell.cell_type != "code":
                    continue

                if self.is_import_only_cell(orig_cell.source):
                    continue
                if self.is_pip_install_only_cell(orig_cell.source):
                    continue

                self.results['summary']['total_cells'] += 1
                original_output = self.original_outputs[idx]

                cell_result = {
                    'cell_number': idx,
                    'reproducible': True,
                    'is_random': False,
                    'source': orig_cell.source,
                    'had_original_output': original_output['has_output']
                }

                # Check for random operations
                is_random = any(pattern in orig_cell.source for pattern in ['random', 'np.random', 'numpy.random'])
                if is_random:
                    cell_result['is_random'] = True
                    self.results['summary']['random_cells'] += 1

                    new_outputs = [self._format_output(out) for out in exec_cell.outputs] if hasattr(exec_cell, 'outputs') else []

                    if not original_output['has_output']:
                        if new_outputs:
                            cell_result['reproducible'] = True
                            cell_result['no_original_output'] = True
                            cell_result['random_execution_successful'] = True
                            self.results['summary']['reproducible_random_cells'] += 1
                        else:
                            # No output on original or rerun - this is fine
                            cell_result['reproducible'] = True
                            cell_result['no_original_output'] = True
                            self.results['summary']['reproducible_random_cells'] += 1
                    else:
                        if not new_outputs:
                            cell_result['reproducible'] = False
                            cell_result['random_execution_failed'] = True
                            self.results['summary']['non_reproducible_random_cells'] += 1
                            self._record_non_reproducible_cell(idx, 'failed_random', {
                                'reason': 'Failed to generate output',
                                'had_original_output': True
                            })
                        else:
                            cell_result['reproducible'] = True
                            cell_result['random_execution_successful'] = True
                            self.results['summary']['reproducible_random_cells'] += 1

                    if cell_result['reproducible']:
                        self.results['summary']['reproducible_cells'] += 1
                    else:
                        self.results['summary']['non_reproducible_cells'] += 1

                else:
                    # Non-random cell handling
                    new_outputs = [self._format_output(out) for out in exec_cell.outputs] if hasattr(exec_cell, 'outputs') else []

                    if not original_output['has_output']:
                        if new_outputs:
                            # Only flag as non-reproducible if we get new outputs on rerun
                            cell_result['has_new_output'] = True
                            cell_result['new_outputs'] = new_outputs
                            cell_result['reproducible'] = False
                            self._record_non_reproducible_cell(idx, 'unexpected_output', new_outputs)
                            self.results['summary']['non_reproducible_cells'] += 1
                        else:
                            # No output in original or rerun - this is fine
                            cell_result['reproducible'] = True
                            self.results['summary']['reproducible_cells'] += 1
                    else:
                        if not new_outputs:
                            cell_result['reproducible'] = False
                            cell_result['missing_expected_output'] = True
                            self.results['summary']['non_reproducible_cells'] += 1
                            self._record_non_reproducible_cell(idx, 'missing_output', original_output['outputs'])
                        else:
                            outputs_match = self._compare_outputs(original_output['outputs'], new_outputs)
                            if not outputs_match:
                                cell_result['reproducible'] = False
                                cell_result['output_mismatch'] = True
                                cell_result['differences'] = {
                                    'original': original_output['outputs'],
                                    'new': new_outputs
                                }
                                self.results['summary']['non_reproducible_cells'] += 1
                                self._record_non_reproducible_cell(idx, 'output_mismatch', {
                                    'original': original_output['outputs'],
                                    'new': new_outputs
                                })
                            else:
                                self.results['summary']['reproducible_cells'] += 1

                self.results['cell_results'].append(cell_result)

        except Exception as e:
            self._record_non_reproducible_cell(idx, 'execution_error', str(e))
            print(f"Error executing notebook: {str(e)}")
            raise

    def generate_non_reproducible_report(self):
        """Generate a detailed report of non-reproducible cells."""
        report = {
            'summary': {
                'total_non_reproducible': self.results['summary']['non_reproducible_cells'],
                'by_category': {
                    'execution_errors': len(self.results['non_reproducible_details']['execution_errors']),
                    'output_mismatches': len(self.results['non_reproducible_details']['output_mismatches']),
                    'missing_outputs': len(self.results['non_reproducible_details']['missing_expected_outputs']),
                    'unexpected_outputs': len(self.results['non_reproducible_details']['unexpected_new_outputs']),
                    'failed_random_cells': len(self.results['non_reproducible_details']['failed_random_cells'])
                }
            },
            'details': self.results['non_reproducible_details']
        }
        return report
    def _analyze_original_outputs(self):
        """Analyze and store the original outputs of each cell."""
        original_outputs = []
        for idx, cell in enumerate(self.notebook.cells):
            if cell.cell_type != "code":
                original_outputs.append(None)
                continue

            output_info = {
                'cell_number': idx,
                'has_output': False,
                'outputs': [],
                'output_types': set()
            }

            if hasattr(cell, 'outputs') and cell.outputs:
                output_info['has_output'] = True
                for output in cell.outputs:
                    output_data = self._format_output(output)
                    output_info['outputs'].append(output_data)
                    output_info['output_types'].add(output_data['type'])

            original_outputs.append(output_info)
        return original_outputs

    def get_package_dependencies(package_name):
        """Get all dependencies for a package including transitive dependencies."""
        try:
            package = pkg_resources.working_set.by_key[package_name]
            deps = {req.key for req in package.requires()}
            return deps
        except:
            return set()

    def analyze_library_dependencies(self):
        """Analyze library dependencies by checking pip install commands in notebook."""
        required_packages = set()
        pip_installed_packages = set()

        package_aliases = {
            'sklearn': 'scikit-learn',
            'scikit-learn': 'scikit-learn',
            'PIL': 'pillow',
            'cv2': 'opencv-python'
        }

        for cell in self.notebook.cells:
            if cell.cell_type != "code":
                continue

            # Check for pip install commands
            pip_pattern = r'!pip\s+install\s+([\w\-\.]+)'
            pip_matches = re.finditer(pip_pattern, cell.source)
            for match in pip_matches:
                package_name = match.group(1)
                if package_name in package_aliases:
                    package_name = package_aliases[package_name]
                pip_installed_packages.add(package_name)

            # Parse imports
            try:
                tree = ast.parse(cell.source)
                for node in ast.walk(tree):
                    if isinstance(node, (ast.Import, ast.ImportFrom)):
                        module_name = ''
                        if isinstance(node, ast.Import):
                            for name in node.names:
                                module_name = name.name.split('.')[0]
                        else:  # ImportFrom
                            if node.module:
                                module_name = node.module.split('.')[0]
                        
                        if module_name:
                            if module_name in package_aliases:
                                module_name = package_aliases[module_name]
                            if module_name not in ('os', 'sys', 'datetime', 'json'):
                                required_packages.add(module_name)
            except:
                continue

        missing_packages = required_packages - pip_installed_packages

        self.results['library_analysis'].update({
            'installed': list(pip_installed_packages),
            'required': list(required_packages),
            'missing': list(missing_packages)
        })

    def _format_output(self, output):
        """Format cell output for comparison."""
        if output.get('output_type') == 'execute_result':
            return {
                'type': 'execute_result',
                'data': output.get('data', {}),
                'execution_count': output.get('execution_count')
            }
        elif output.get('output_type') == 'stream':
            return {
                'type': 'stream',
                'name': output.get('name'),
                'text': output.get('text')
            }
        elif output.get('output_type') == 'display_data':
            return {
                'type': 'display_data',
                'data': output.get('data', {})
            }
        return output

    def analyze_notebook(self):
        """Main analysis method."""
        self.analyze_library_dependencies()
        self.execute_and_compare_cells()
        self.analyze_ml_models()
        return self.results

    def _compare_outputs(self, original_outputs, new_outputs):
        """Compare two sets of outputs for equality."""
        if len(original_outputs) != len(new_outputs):
            return False

        for orig, new in zip(original_outputs, new_outputs):
            if orig['type'] != new['type']:
                return False

            if orig['type'] == 'execute_result':
                if orig['data'] != new['data']:
                    return False
            elif orig['type'] == 'stream':
                if orig['text'] != new['text']:
                    return False
            elif orig['type'] == 'display_data':
                if orig['data'] != new['data']:
                    return False

        return True

    def analyze_ml_models(self):
        """Analyze machine learning models in the notebook for bias and feature importance."""
        namespace = self.base_namespace.copy()

        for idx, cell in enumerate(self.notebook.cells):
            if cell.cell_type != "code":
                continue

            if self.is_pip_install_only_cell(cell.source):
                continue

            try:
                # Execute this cell in the accumulated namespace
                exec(cell.source, namespace)

                # Find model variable
                model_var = None
                for var_name, var_val in namespace.items():
                    if (hasattr(var_val, 'fit') and hasattr(var_val, 'predict') and
                        not var_name.startswith('_') and
                        var_name not in self.base_namespace):
                        model_var = var_name
                        model = var_val
                        break

                if model_var and 'X_test' in namespace and 'y_test' in namespace:
                    print(f"Found model {model_var} in cell {idx}")
                    # Calculate original model bias
                    orig_preds = model.predict(namespace['X_test'])
                    orig_bias = self._calculate_bias(orig_preds, namespace['y_test'])
                    print(f'orig_bias: {orig_bias}')

                    # Store original model and data
                    model_info = {
                        'cell_number': idx,
                        'variable_name': model_var,
                        'original_bias': orig_bias,
                        'model_type': type(model).__name__,
                        'parameters': model.get_params()
                    }

                    # Retrain model with same parameters
                    new_model = model.__class__(**model.get_params())
                    new_model.fit(namespace['X_train'], namespace['y_train'])
                    new_preds = new_model.predict(namespace['X_test'])
                    new_bias = self._calculate_bias(new_preds, namespace['y_test'])
                    print(f"new_bias: {new_bias}")

                    model_info['retrained_bias'] = new_bias
                    bias_difference = abs(orig_bias - new_bias)
                    model_info['bias_difference'] = bias_difference

                    # Mark cell as non-reproducible if bias difference exceeds threshold
                    if bias_difference > self.bias_threshold:
                        # Update the cell results to mark it as non-reproducible
                        cell_result = {
                            'cell_number': idx,
                            'reproducible': False,
                            'source': cell.source,
                            'reason': f'Model bias difference {bias_difference:.3f} exceeds threshold {self.bias_threshold}',
                            'is_model': True,
                            'bias_difference': bias_difference
                        }

                        # Add to non-reproducible details
                        self._record_non_reproducible_cell(idx, 'model_bias_variation', {
                            'original_bias': orig_bias,
                            'retrained_bias': new_bias,
                            'bias_difference': bias_difference,
                            'threshold': self.bias_threshold
                        })

                        # Update summary statistics
                        self.results['summary']['non_reproducible_cells'] += 1

                        # Perform LIME analysis for explanation
                        lime_analysis = self._perform_lime_analysis(
                            model,
                            new_model,
                            namespace['X_train'],
                            namespace['X_test']
                        )
                        model_info['lime_analysis'] = lime_analysis

                    self.results['ml_analysis']['models'][f'model_{idx}'] = model_info
                    print(f"Successfully analyzed model in cell {idx}")

            except Exception as e:
                print(f"Error analyzing model in cell {idx}: {str(e)}")
                continue


    def _calculate_bias(self, predictions, actual):
        """Calculate bias using statistical parity difference"""
        unique_classes = np.unique(actual)
        class_predictions = [predictions == c for c in unique_classes]
        class_rates = [np.mean(pred) for pred in class_predictions]

        # Statistical parity difference between classes
        bias = max(class_rates) - min(class_rates)
        return bias

    def _perform_lime_analysis(self, original_model, retrained_model, X_train, X_test, num_features=10):
        """Perform LIME analysis on both original and retrained models."""
        try:
            # Create LIME explainer
            explainer = lime_tabular.LimeTabularExplainer(
                training_data=X_train.values if hasattr(X_train, 'values') else X_train,
                feature_names=X_train.columns if hasattr(X_train, 'columns') else None,
                mode='classification'
            )

            # Get explanations for a sample instance
            sample_instance = X_test.iloc[0] if hasattr(X_test, 'iloc') else X_test[0]

            # Generate explanations for both models
            orig_exp = explainer.explain_instance(
                sample_instance,
                original_model.predict_proba,
                num_features=num_features
            )

            new_exp = explainer.explain_instance(
                sample_instance,
                retrained_model.predict_proba,
                num_features=num_features
            )

            # Compare feature importance differences
            feature_diffs = self._compare_lime_explanations(
                orig_exp.as_list(),
                new_exp.as_list()
            )

            return {
                'original_explanation': orig_exp.as_list(),
                'retrained_explanation': new_exp.as_list(),
                'feature_importance_differences': feature_diffs
            }

        except Exception as e:
            print(f"Error in LIME analysis: {str(e)}")
            return None

    def _compare_lime_explanations(self, orig_exp, new_exp):
        """Compare LIME explanations between original and retrained models."""
        orig_dict = dict(orig_exp)
        new_dict = dict(new_exp)

        differences = {}
        all_features = set(orig_dict.keys()) | set(new_dict.keys())

        for feature in all_features:
            orig_val = orig_dict.get(feature, 0)
            new_val = new_dict.get(feature, 0)
            diff = abs(orig_val - new_val)

            if diff > 0.01:  # Threshold for significant difference
                differences[feature] = {
                    'original_importance': orig_val,
                    'retrained_importance': new_val,
                    'absolute_difference': diff
                }

        return differences
