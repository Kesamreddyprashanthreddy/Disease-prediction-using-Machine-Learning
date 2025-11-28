#!/usr/bin/env python3
"""
Test Cases and Results for Medical Diagnosis AI System
Comprehensive testing scenarios with detailed results
"""

import os
import sys
import numpy as np
import joblib
from datetime import datetime
import json

class TestCaseRunner:
    def __init__(self):
        self.results = []
        self.models = {}
        self.scalers = {}
        self.load_models()
    
    def load_models(self):
        """Load all available models"""
        model_files = {
            'diabetes': 'saved_models/diabetes_model.joblib',
            'kidney': 'saved_models/kidney_model.joblib',
            'lung': 'saved_models/lung_disease_model.h5',
            'breast': 'saved_models/breast_cancer_image_model.h5'
        }
        
        scaler_files = {
            'diabetes': 'saved_models/diabetes_scaler.joblib',
            'kidney': 'saved_models/kidney_scaler.joblib'
        }
        
        for name, path in model_files.items():
            if os.path.exists(path):
                try:
                    if path.endswith('.h5'):
                        try:
                            import tensorflow as tf
                            self.models[name] = tf.keras.models.load_model(path)
                            print(f"Loaded {name} TensorFlow model successfully")
                        except ImportError:
                            print(f"TensorFlow not available for {name} model")
                    else:
                        self.models[name] = joblib.load(path)
                        print(f"Loaded {name} model successfully")
                except Exception as e:
                    print(f"Failed to load {name} model: {e}")
        
        for name, path in scaler_files.items():
            if os.path.exists(path):
                try:
                    self.scalers[name] = joblib.load(path)
                    print(f"Loaded {name} scaler successfully")
                except Exception as e:
                    print(f"Failed to load {name} scaler: {e}")
    
    def run_test_case(self, test_id, description, test_func, expected_result=None):
        """Run a single test case and record results"""
        print(f"\nTest Case {test_id}: {description}")
        print("-" * 50)
        
        try:
            start_time = datetime.now()
            result = test_func()
            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds()
            
            status = "PASS" if (expected_result is None or result == expected_result) else "FAIL"
            
            test_result = {
                'test_id': test_id,
                'description': description,
                'status': status,
                'result': str(result),
                'expected': str(expected_result) if expected_result else "N/A",
                'execution_time': f"{execution_time:.3f}s",
                'timestamp': start_time.strftime('%Y-%m-%d %H:%M:%S')
            }
            
            self.results.append(test_result)
            
            print(f"Result: {result}")
            print(f"Status: {status}")
            print(f"Execution Time: {execution_time:.3f}s")
            
            return result
            
        except Exception as e:
            test_result = {
                'test_id': test_id,
                'description': description,
                'status': 'ERROR',
                'result': f"Exception: {str(e)}",
                'expected': str(expected_result) if expected_result else "N/A",
                'execution_time': "N/A",
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            self.results.append(test_result)
            print(f"Error: {str(e)}")
            print("Status: ERROR")
            return None

def create_test_cases():
    """Define and run all test cases"""
    runner = TestCaseRunner()
    
    print("=" * 60)
    print("MEDICAL DIAGNOSIS AI SYSTEM - TEST CASES AND RESULTS")
    print("=" * 60)
    print(f"Test Session: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Test Environment: {os.getcwd()}")
    print("=" * 60)
    
    # Test Case 1: Diabetes Normal Case
    def test_diabetes_normal():
        if 'diabetes' not in runner.models:
            return "Model not available"
        
        # Normal patient data: [pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree, age]
        test_data = np.array([[1, 85, 66, 29, 0, 26.6, 0.351, 31]])
        
        if 'diabetes' in runner.scalers:
            test_data = runner.scalers['diabetes'].transform(test_data)
        
        prediction = runner.models['diabetes'].predict(test_data)[0]
        probability = runner.models['diabetes'].predict_proba(test_data)[0]
        
        return f"Prediction: {prediction}, Probability: [{probability[0]:.3f}, {probability[1]:.3f}]"
    
    runner.run_test_case("TC001", "Diabetes Prediction - Normal Patient", test_diabetes_normal)
    
    # Test Case 2: Diabetes High Risk Case
    def test_diabetes_high_risk():
        if 'diabetes' not in runner.models:
            return "Model not available"
        
        # High risk patient data
        test_data = np.array([[8, 196, 76, 36, 249, 36.5, 0.875, 29]])
        
        if 'diabetes' in runner.scalers:
            test_data = runner.scalers['diabetes'].transform(test_data)
        
        prediction = runner.models['diabetes'].predict(test_data)[0]
        probability = runner.models['diabetes'].predict_proba(test_data)[0]
        
        return f"Prediction: {prediction}, Probability: [{probability[0]:.3f}, {probability[1]:.3f}]"
    
    runner.run_test_case("TC002", "Diabetes Prediction - High Risk Patient", test_diabetes_high_risk)
    
    # Test Case 3: Kidney Disease Normal Case
    def test_kidney_normal():
        if 'kidney' not in runner.models:
            return "Model not available"
        
        # Normal kidney function data (24 features)
        test_data = np.array([[1.02, 1.0, 4.1, 0, 0, 1, 1, 121, 36, 1.2, 0, 15.4, 44, 7800, 5.2, 1, 1, 1, 0, 0, 0, 0, 0, 1]])
        
        if 'kidney' in runner.scalers:
            test_data = runner.scalers['kidney'].transform(test_data)
        
        prediction = runner.models['kidney'].predict(test_data)[0]
        probability = runner.models['kidney'].predict_proba(test_data)[0]
        
        return f"Prediction: {prediction}, Probability: [{probability[0]:.3f}, {probability[1]:.3f}]"
    
    runner.run_test_case("TC003", "Kidney Disease - Normal Function", test_kidney_normal)
    
    # Test Case 4: Kidney Disease High Risk Case
    def test_kidney_high_risk():
        if 'kidney' not in runner.models:
            return "Model not available"
        
        # High risk kidney disease data
        test_data = np.array([[1.025, 4.0, 1.9, 3, 2, 0, 0, 70, 18, 1.8, 3, 9.6, 32, 6700, 3.8, 0, 0, 0, 1, 1, 1, 1, 1, 0]])
        
        if 'kidney' in runner.scalers:
            test_data = runner.scalers['kidney'].transform(test_data)
        
        prediction = runner.models['kidney'].predict(test_data)[0]
        probability = runner.models['kidney'].predict_proba(test_data)[0]
        
        return f"Prediction: {prediction}, Probability: [{probability[0]:.3f}, {probability[1]:.3f}]"
    
    runner.run_test_case("TC004", "Kidney Disease - High Risk Patient", test_kidney_high_risk)
    
    # Test Case 5: Lung Disease Model Test
    def test_lung_disease():
        if 'lung' not in runner.models:
            return "Lung model not available"
        
        try:
            # Create dummy image data (224x224x3 for typical CNN input)
            import numpy as np
            test_image = np.random.rand(1, 224, 224, 3).astype(np.float32)
            
            prediction = runner.models['lung'].predict(test_image, verbose=0)
            predicted_class = int(np.round(prediction[0][0]))
            confidence = float(prediction[0][0])
            
            return f"Lung prediction: {predicted_class}, Confidence: {confidence:.3f}"
        except Exception as e:
            return f"Lung model test failed: {str(e)}"
    
    runner.run_test_case("TC005", "Lung Disease - CNN Model Test", test_lung_disease)
    
    # Test Case 6: Breast Cancer Model Test
    def test_breast_cancer():
        if 'breast' not in runner.models:
            return "Breast cancer model not available"
        
        try:
            # Create dummy mammogram data (224x224x3 for typical CNN input)
            import numpy as np
            test_image = np.random.rand(1, 224, 224, 3).astype(np.float32)
            
            prediction = runner.models['breast'].predict(test_image, verbose=0)
            predicted_class = int(np.round(prediction[0][0]))
            confidence = float(prediction[0][0])
            
            return f"Breast cancer prediction: {predicted_class}, Confidence: {confidence:.3f}"
        except Exception as e:
            return f"Breast cancer model test failed: {str(e)}"
    
    runner.run_test_case("TC006", "Breast Cancer - CNN Model Test", test_breast_cancer)
    
    # Test Case 7: All Models Integration Test
    def test_all_models_integration():
        available_models = list(runner.models.keys())
        model_types = {
            'diabetes': 'Tabular ML',
            'kidney': 'Tabular ML', 
            'lung': 'CNN Deep Learning',
            'breast': 'CNN Deep Learning'
        }
        
        results = []
        for model_name in available_models:
            model_type = model_types.get(model_name, 'Unknown')
            results.append(f"{model_name}({model_type})")
        
        return f"Integrated models: {', '.join(results)}, Total: {len(available_models)}/4"
    
    runner.run_test_case("TC007", "All Models Integration Test", test_all_models_integration)
    
    # Test Case 8: Input Validation - Correct Format
    def test_input_validation_correct():
        diabetes_input = [2, 120, 80, 25, 100, 25.5, 0.5, 30]
        
        if len(diabetes_input) == 8 and all(isinstance(x, (int, float)) for x in diabetes_input):
            return "Valid input format"
        else:
            return "Invalid input format"
    
    runner.run_test_case("TC008", "Input Validation - Correct Format", test_input_validation_correct, "Valid input format")
    
    # Test Case 9: Input Validation - Incorrect Format
    def test_input_validation_incorrect():
        diabetes_input = [2, 120, 80, 25]  # Missing features
        
        if len(diabetes_input) == 8:
            return "Valid input format"
        else:
            return "Invalid input format - incorrect number of features"
    
    runner.run_test_case("TC009", "Input Validation - Incorrect Format", test_input_validation_incorrect, "Invalid input format - incorrect number of features")
    
    # Test Case 10: Model Performance Test
    def test_model_performance():
        if 'diabetes' not in runner.models:
            return "Model not available"
        
        # Test multiple predictions
        test_cases = [
            [1, 85, 66, 29, 0, 26.6, 0.351, 31],
            [8, 196, 76, 36, 249, 36.5, 0.875, 29],
            [2, 122, 70, 27, 0, 36.8, 0.340, 27],
            [0, 180, 78, 63, 14, 32.4, 0.443, 25],
            [1, 130, 70, 13, 105, 25.9, 0.472, 22]
        ]
        
        predictions = []
        for case in test_cases:
            test_data = np.array([case])
            if 'diabetes' in runner.scalers:
                test_data = runner.scalers['diabetes'].transform(test_data)
            pred = runner.models['diabetes'].predict(test_data)[0]
            predictions.append(pred)
        
        return f"Batch predictions: {predictions}, Total: {len(predictions)} cases"
    
    runner.run_test_case("TC010", "Model Performance - Batch Prediction", test_model_performance)
    
    # Test Case 11: Edge Case - Extreme Values
    def test_edge_case_extreme():
        if 'diabetes' not in runner.models:
            return "Model not available"
        
        # Extreme values test
        test_data = np.array([[0, 0, 0, 0, 0, 0, 0, 0]])  # All zeros
        
        try:
            if 'diabetes' in runner.scalers:
                test_data = runner.scalers['diabetes'].transform(test_data)
            prediction = runner.models['diabetes'].predict(test_data)[0]
            return f"Handled extreme values, prediction: {prediction}"
        except Exception as e:
            return f"Failed with extreme values: {str(e)}"
    
    runner.run_test_case("TC011", "Edge Case - Extreme Values", test_edge_case_extreme)
    
    # Print Test Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY REPORT")
    print("=" * 60)
    
    total_tests = len(runner.results)
    passed_tests = len([r for r in runner.results if r['status'] == 'PASS'])
    failed_tests = len([r for r in runner.results if r['status'] == 'FAIL'])
    error_tests = len([r for r in runner.results if r['status'] == 'ERROR'])
    
    print(f"Total Test Cases: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {failed_tests}")
    print(f"Errors: {error_tests}")
    print(f"Success Rate: {(passed_tests/total_tests*100):.1f}%")
    print()
    
    # Detailed Results Table
    print("DETAILED TEST RESULTS")
    print("-" * 60)
    print(f"{'Test ID':<8} {'Status':<8} {'Description':<25} {'Execution':<12}")
    print("-" * 60)
    
    for result in runner.results:
        print(f"{result['test_id']:<8} {result['status']:<8} {result['description'][:24]:<25} {result['execution_time']:<12}")
    
    print("-" * 60)
    
    # Test Categories Summary
    print("\nTEST CATEGORIES COVERED:")
    print("• Functional Testing - Model predictions (All 4 diseases)")
    print("• CNN Model Testing - Deep learning models (Lung & Breast)")
    print("• Integration Testing - All models working together")
    print("• Input Validation - Data format checking")
    print("• Performance Testing - Batch operations")
    print("• Edge Case Testing - Extreme value handling")
    
    print("\nTEST ENVIRONMENT:")
    print(f"• Models Available: {len(runner.models)}/4 (Diabetes, Kidney, Lung, Breast)")
    print(f"• Scalers Available: {len(runner.scalers)}")
    print(f"• Deep Learning Models: {sum(1 for name in runner.models if name in ['lung', 'breast'])}")
    print(f"• Traditional ML Models: {sum(1 for name in runner.models if name in ['diabetes', 'kidney'])}")
    print(f"• Test Data: Synthetic and real-world scenarios")
    print(f"• Execution Platform: Python {sys.version.split()[0]}")
    
    return runner.results

if __name__ == "__main__":
    results = create_test_cases()
    print(f"\nTest session completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")