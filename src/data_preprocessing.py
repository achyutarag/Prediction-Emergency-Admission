"""
Data Preprocessing Module for Healthcare ML Prediction

This module contains functions for data cleaning, feature engineering,
and preprocessing of emergency department patient data.

Author: Achyuta Raghunathan
Course: UC Irvine Math 10, Spring 2025
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')


def load_and_clean_data(file_path):
    """
    Load and perform initial cleaning of the healthcare dataset.
    
    Parameters:
    -----------
    file_path : str
        Path to the CSV file containing the healthcare data
        
    Returns:
    --------
    pd.DataFrame
        Cleaned dataframe ready for preprocessing
    """
    try:
        df = pd.read_csv(file_path)
        print(f"✅ Successfully loaded dataset with {df.shape[0]} rows and {df.shape[1]} columns")
        return df
    except FileNotFoundError:
        print(f"❌ Error: File not found at {file_path}")
        return None
    except Exception as e:
        print(f"❌ Error loading data: {str(e)}")
        return None


def expand_summary_to_patient_level(df):
    """
    Convert summary table format to patient-level observations.
    
    This function transforms the original summary table structure
    into individual patient records suitable for machine learning.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Summary table with aggregated statistics
        
    Returns:
    --------
    pd.DataFrame
        Patient-level dataset with individual observations
    """
    print("🔄 Converting summary table to patient-level data...")
    
    # Calculate total count of ED visits
    total_count = int(df.loc[df["Parameter"] == "N", "<$42 999"].sum() +
                      df.loc[df["Parameter"] == "N", "$43 000-$53 999"].sum() +
                      df.loc[df["Parameter"] == "N", "$54 000-$70 999"].sum() +
                      df.loc[df["Parameter"] == "N", ">$71 000"].sum())
    
    # Initialize dictionary for patient-level features
    features = {
        "Mean Age": [],
        "Sex": [],
        "Location": [],
        "Region": [],
        "Teaching": [],
        "Payer": [],
        "Diagnosis": [],
        "Label": []
    }
    
    # Expand Mean Age
    mean_age_value = df.loc[df["Parameter"] == "Mean age in years", "Total"].values[0]
    features["Mean Age"] = [mean_age_value] * total_count
    
    # Binary Label: Admitted = 1, Not Admitted = 0
    features["Label"] = [1]*15743 + [0]*17974  # Based on dataset breakdown
    
    # Function to expand categorical features
    def expand_feature(prefix, feature_name):
        rows = df[df["Parameter"].str.startswith(prefix)]
        for _, row in rows.iterrows():
            value = row["Parameter"].replace(prefix, "").strip()
            count = int(row["Total"])
            features[feature_name].extend([value] * count)
    
    # Expand all categorical features
    expand_feature("Sex:", "Sex")
    expand_feature("Location:", "Location")
    expand_feature("Region:", "Region")
    expand_feature("Teaching:", "Teaching")
    expand_feature("Payer:", "Payer")
    
    # Expand Diagnosis (pad with 'None' for missing entries)
    diagnosis_rows = df[df["Parameter"].str.startswith("Dx:")]
    diagnosis_total = 0
    for _, row in diagnosis_rows.iterrows():
        value = row["Parameter"].replace("Dx:", "").strip()
        count = int(row["Total"])
        diagnosis_total += count
        features["Diagnosis"].extend([value] * count)
    features["Diagnosis"].extend(["None"] * (total_count - diagnosis_total))
    
    # Align feature lengths
    min_len = min(len(lst) for lst in features.values())
    for key in features:
        features[key] = features[key][:min_len]
    
    # Create DataFrame
    df_obs = pd.DataFrame(features)
    print(f"✅ Created patient-level dataset with {df_obs.shape[0]} observations")
    
    return df_obs


def create_preprocessing_pipeline():
    """
    Create a preprocessing pipeline for the healthcare dataset.
    
    Returns:
    --------
    sklearn.pipeline.Pipeline
        Preprocessing pipeline with scaling and encoding
    """
    # Define categorical and numeric features
    categorical_features = ["Sex", "Location", "Region", "Teaching", "Payer", "Diagnosis"]
    numeric_features = ["Mean Age"]
    
    # Create preprocessing pipeline
    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(drop="first"), categorical_features)
    ])
    
    pipeline = Pipeline([("preprocessor", preprocessor)])
    
    print("✅ Created preprocessing pipeline with StandardScaler and OneHotEncoder")
    return pipeline


def preprocess_data(df, pipeline):
    """
    Apply preprocessing pipeline to the dataset.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Raw patient-level dataset
    pipeline : sklearn.pipeline.Pipeline
        Preprocessing pipeline
        
    Returns:
    --------
    tuple
        (X_transformed, y, feature_names) where:
        - X_transformed: Preprocessed feature matrix
        - y: Target labels
        - feature_names: Names of transformed features
    """
    print("🔄 Applying preprocessing pipeline...")
    
    # Separate features and target
    X = df.drop(columns=["Label"])
    y = df["Label"]
    
    # Apply preprocessing
    X_transformed = pipeline.fit_transform(X)
    
    # Get feature names
    feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out()
    
    print(f"✅ Preprocessing complete. Shape: {X_transformed.shape}")
    print(f"✅ Feature names generated: {len(feature_names)} features")
    
    return X_transformed, y, feature_names


def analyze_feature_correlations(X_transformed, feature_names, threshold=0.8):
    """
    Analyze feature correlations and identify highly correlated pairs.
    
    Parameters:
    -----------
    X_transformed : array-like
        Preprocessed feature matrix
    feature_names : list
        Names of features
    threshold : float, default=0.8
        Correlation threshold for identifying highly correlated features
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with highly correlated feature pairs
    """
    print(f"🔍 Analyzing feature correlations (threshold: {threshold})...")
    
    # Create DataFrame with feature names
    df_features = pd.DataFrame(X_transformed.toarray() if hasattr(X_transformed, "toarray") else X_transformed,
                              columns=feature_names)
    
    # Calculate correlation matrix
    corr_matrix = df_features.corr()
    
    # Find highly correlated pairs
    strong_corrs = (
        corr_matrix.where(~np.eye(corr_matrix.shape[0], dtype=bool))
        .stack()
        .reset_index()
    )
    strong_corrs.columns = ["Feature 1", "Feature 2", "Correlation"]
    strong_corrs_filtered = strong_corrs[strong_corrs["Correlation"].abs() > threshold]
    strong_corrs_filtered = strong_corrs_filtered.sort_values(by="Correlation", key=np.abs, ascending=False)
    
    print(f"✅ Found {len(strong_corrs_filtered)} highly correlated feature pairs")
    
    return strong_corrs_filtered, corr_matrix


def save_processed_data(X_transformed, y, feature_names, output_path):
    """
    Save processed data to CSV file.
    
    Parameters:
    -----------
    X_transformed : array-like
        Preprocessed feature matrix
    y : array-like
        Target labels
    feature_names : list
        Feature names
    output_path : str
        Path to save the processed data
    """
    print(f"💾 Saving processed data to {output_path}...")
    
    # Create DataFrame
    df_processed = pd.DataFrame(X_transformed.toarray() if hasattr(X_transformed, "toarray") else X_transformed,
                               columns=feature_names)
    df_processed["Label"] = y.reset_index(drop=True)
    
    # Save to CSV
    df_processed.to_csv(output_path, index=False)
    
    print(f"✅ Processed data saved successfully")
    print(f"   - Shape: {df_processed.shape}")
    print(f"   - Features: {len(feature_names)}")
    print(f"   - Target distribution: {y.value_counts().to_dict()}")


def main():
    """
    Main function to demonstrate the preprocessing pipeline.
    """
    print("🏥 Healthcare ML Data Preprocessing Pipeline")
    print("=" * 50)
    
    # Example usage (uncomment when you have the actual data file)
    # df = load_and_clean_data("data/raw/hypertensive_crisis_data.csv")
    # df_obs = expand_summary_to_patient_level(df)
    # pipeline = create_preprocessing_pipeline()
    # X_transformed, y, feature_names = preprocess_data(df_obs, pipeline)
    # save_processed_data(X_transformed, y, feature_names, "data/processed/preprocessed_data.csv")
    
    print("✅ Preprocessing pipeline ready for use!")


if __name__ == "__main__":
    main()
