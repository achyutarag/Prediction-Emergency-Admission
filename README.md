# 🏥 Healthcare ML: Emergency Department Disposition Prediction

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org)

## 📋 Project Overview

This project develops machine learning models to predict emergency department (ED) disposition outcomes for patients presenting with hypertensive crises. Using clinical and socioeconomic indicators, we achieve **99.99% accuracy** in predicting whether patients will be admitted to the hospital or discharged.

###  Key Results
- **Model Performance**: 99.99% accuracy with both Logistic Regression and Random Forest
- **Dataset**: 33,727 weighted ED visits from Nationwide Emergency Department Sample (NEDS)
- **Features**: 100+ features including demographics, payer type, hospital characteristics
- **Business Impact**: Potential to improve ED triage efficiency and reduce healthcare disparities

##  Quick Start

### Prerequisites
```bash
Python 3.8+
Jupyter Notebook
```

### Installation
```bash
git clone https://github.com/yourusername/healthcare-ml-prediction.git
cd healthcare-ml-prediction
pip install -r requirements.txt
```

### Running the Analysis
```bash
# Start Jupyter Notebook
jupyter notebook

# Navigate to notebooks/ folder and run in order:
# 1. 1_preprocessing_scaling.ipynb
# 2. 2_binary_logistic_regression.ipynb  
# 3. 3_random_forest.ipynb
# 4. 4_discussion_insights.ipynb
```


##  Methodology

### Data Source
- **Dataset**: Nationwide Emergency Department Sample (NEDS)
- **Study**: Peer-reviewed clinical study on hypertensive crises in U.S. EDs
- **Sample Size**: 33,727 weighted ED visits (from 32M+ filtered to 9,437 visits)
- **Inclusion Criteria**: Age 18-90, valid income quartile, primary diagnosis of hypertensive crisis

### Data Preprocessing
1. **Data Transformation**: Converted summary table to patient-level format
2. **Feature Engineering**: One-hot encoding for categorical variables
3. **Scaling**: StandardScaler for numeric features (age)
4. **Target Encoding**: Binary classification (Admitted=1, Not Admitted=0)

### Models Implemented
1. **Logistic Regression**: Linear model for interpretable predictions
2. **Random Forest**: Ensemble method for robust classification
3. **Cross-Validation**: 5-fold stratified CV for model validation

## 📊 Key Findings

### Model Performance
| Metric | Logistic Regression | Random Forest |
|--------|-------------------|---------------|
| **Accuracy** | 99.99% | 99.99% |
| **Precision** | 1.00 | 1.00 |
| **Recall** | 1.00 | 1.00 |
| **F1-Score** | 1.00 | 1.00 |
| **ROC AUC** | 0.9999 | 1.00 |

### Top Predictors
1. **Payer_Medicare** (Most influential) - Medicare patients more likely to be admitted
2. **Diagnosis_None** (Negative predictor) - Missing diagnosis reduces admission likelihood
3. **Sex_Male** - Male patients slightly more likely to be admitted
4. **Geographic factors** - Hospital location and region significantly impact decisions

##  Business Impact

### Clinical Applications
- **ED Triage Optimization**: Improve patient flow and resource allocation
- **Risk Stratification**: Identify high-risk patients requiring immediate attention
- **Quality Assurance**: Monitor admission decision consistency across providers

### Healthcare Equity
- **Bias Detection**: Identify potential socioeconomic biases in admission decisions
- **Policy Development**: Inform evidence-based healthcare policies
- **Resource Planning**: Optimize hospital capacity and staffing

##  Technical Stack

- **Python 3.8+**
- **Scikit-learn**: Machine learning algorithms
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing
- **Matplotlib/Seaborn**: Data visualization
- **Jupyter**: Interactive development environment

##  Future Work

- [ ] **Deep Learning Models**: Implement neural networks for comparison
- [ ] **Real-time Deployment**: Deploy model in production ED systems
- [ ] **Feature Engineering**: Explore additional clinical indicators
- [ ] **Fairness Analysis**: Comprehensive bias detection and mitigation
- [ ] **Multi-class Extension**: Predict specific admission types (ICU, general ward, etc.)

##  Author

**Achyuta Raghunathan**  
UC Irvine, Math 10, Spring 2025  
Course Project: Healthcare Machine Learning

##  License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

##  Acknowledgments

- Original research paper: [PMC10024970](https://pmc.ncbi.nlm.nih.gov/articles/PMC10024970/)
- Nationwide Emergency Department Sample (NEDS) dataset
- UC Irvine Math 10 course materials

##  Contact

For questions or collaboration opportunities, please reach out through GitHub issues or email.

---

*This project demonstrates advanced machine learning techniques applied to healthcare data, showcasing skills in data preprocessing, model development, evaluation, and business impact analysis.*
