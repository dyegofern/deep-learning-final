"""
Data Processing Module
Functions for loading, cleaning, and feature engineering aviation emissions data.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from typing import Tuple, Dict


def generate_synthetic_aviation_data(n_samples: int = 5000, random_state: int = 42) -> pd.DataFrame:
    """
    Generate simulated aviation emissions data for demonstration.
    Replace this with real QAR data loading when available.

    Parameters:
    -----------
    n_samples : int
        Number of samples to generate
    random_state : int
        Random seed for reproducibility

    Returns:
    --------
    DataFrame : Synthetic aviation emissions dataset
    """
    np.random.seed(random_state)

    # Aircraft types with different base emissions
    aircraft_types = ['A320', 'A321', 'B737', 'B738', 'B777', 'B787', 'A380']
    aircraft_weights = {'A320': 77, 'A321': 93, 'B737': 79, 'B738': 85,
                       'B777': 350, 'B787': 250, 'A380': 575}

    # Flight phases
    phases = ['taxi', 'climb', 'cruise', 'descent', 'approach']
    phase_fuel_multipliers = {'taxi': 0.3, 'climb': 2.5, 'cruise': 1.0,
                             'descent': 0.4, 'approach': 0.6}

    data = []

    for _ in range(n_samples):
        aircraft = np.random.choice(aircraft_types, p=[0.25, 0.20, 0.20, 0.15, 0.08, 0.07, 0.05])
        phase = np.random.choice(phases)

        # Generate features based on phase
        if phase == 'taxi':
            altitude = np.random.uniform(0, 100)
            speed = np.random.uniform(5, 30)
        elif phase == 'climb':
            altitude = np.random.uniform(1000, 35000)
            speed = np.random.uniform(200, 350)
        elif phase == 'cruise':
            altitude = np.random.uniform(30000, 42000)
            speed = np.random.uniform(420, 520)
        elif phase == 'descent':
            altitude = np.random.uniform(5000, 30000)
            speed = np.random.uniform(250, 350)
        else:  # approach
            altitude = np.random.uniform(500, 5000)
            speed = np.random.uniform(120, 200)

        # Weight varies by aircraft type
        base_weight = aircraft_weights[aircraft]
        weight = np.random.uniform(base_weight * 0.6, base_weight * 1.2)

        # Route distance
        route_distance = np.random.uniform(100, 6000)

        # Weather
        temperature = np.random.uniform(-40, 30)
        wind_speed = np.random.uniform(-50, 50)  # negative = headwind

        # Calculate CO2 emissions (simplified physics-based model)
        base_fuel = weight * phase_fuel_multipliers[phase] * 0.8
        altitude_factor = 1 - (altitude / 100000)
        speed_factor = 1 + (speed / 1000)
        wind_factor = 1 - (wind_speed / 200)

        co2_kg = base_fuel * altitude_factor * speed_factor * wind_factor
        co2_kg = max(10, co2_kg + np.random.normal(0, co2_kg * 0.1))

        data.append({
            'aircraft_type': aircraft,
            'flight_phase': phase,
            'altitude_ft': altitude,
            'speed_knots': speed,
            'weight_tons': weight,
            'route_distance_nm': route_distance,
            'temperature_c': temperature,
            'wind_speed_knots': wind_speed,
            'co2_kg': co2_kg
        })

    return pd.DataFrame(data)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create engineered features from raw aviation data.

    Parameters:
    -----------
    df : DataFrame
        Raw aviation data

    Returns:
    --------
    DataFrame : Data with engineered features
    """
    df_eng = df.copy()

    # 1. Speed-to-weight ratio (efficiency metric)
    df_eng['speed_weight_ratio'] = df_eng['speed_knots'] / df_eng['weight_tons']

    # 2. Altitude category (binned)
    df_eng['altitude_category'] = pd.cut(df_eng['altitude_ft'],
                                         bins=[0, 5000, 20000, 35000, 50000],
                                         labels=['low', 'medium', 'high', 'very_high'])

    # 3. Is heavy aircraft (weight > 200 tons)
    df_eng['is_heavy'] = (df_eng['weight_tons'] > 200).astype(int)

    # 4. Wind impact (headwind negative, tailwind positive)
    df_eng['wind_impact'] = -df_eng['wind_speed_knots']

    return df_eng


def encode_categorical_features(df: pd.DataFrame,
                                categorical_cols: list = None) -> pd.DataFrame:
    """
    One-hot encode categorical features.

    Parameters:
    -----------
    df : DataFrame
        Data with categorical features
    categorical_cols : list
        List of categorical column names to encode

    Returns:
    --------
    DataFrame : Data with one-hot encoded features
    """
    if categorical_cols is None:
        categorical_cols = ['aircraft_type', 'flight_phase', 'altitude_category']

    df_encoded = pd.get_dummies(df, columns=categorical_cols,
                                prefix=['aircraft', 'phase', 'alt_cat'],
                                drop_first=False, dtype=int)

    return df_encoded


def split_data(X: pd.DataFrame, y: pd.Series,
               test_size: float = 0.15,
               val_size: float = 0.15,
               random_state: int = 42) -> Tuple:
    """
    Split data into train, validation, and test sets.

    Parameters:
    -----------
    X : DataFrame
        Features
    y : Series
        Target variable
    test_size : float
        Proportion for test set
    val_size : float
        Proportion for validation set
    random_state : int
        Random seed

    Returns:
    --------
    tuple : (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    # First split: separate test set
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Second split: separate validation from training
    val_ratio = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_ratio, random_state=random_state
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


def scale_features(X_train: pd.DataFrame,
                   X_val: pd.DataFrame = None,
                   X_test: pd.DataFrame = None) -> Tuple:
    """
    Scale features using StandardScaler.

    Parameters:
    -----------
    X_train : DataFrame
        Training features
    X_val : DataFrame, optional
        Validation features
    X_test : DataFrame, optional
        Test features

    Returns:
    --------
    tuple : (scaler, X_train_scaled, X_val_scaled, X_test_scaled)
    """
    scaler = StandardScaler()

    # Fit on training data
    X_train_scaled = scaler.fit_transform(X_train)
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)

    # Transform validation and test
    X_val_scaled = None
    X_test_scaled = None

    if X_val is not None:
        X_val_scaled = scaler.transform(X_val)
        X_val_scaled = pd.DataFrame(X_val_scaled, columns=X_val.columns, index=X_val.index)

    if X_test is not None:
        X_test_scaled = scaler.transform(X_test)
        X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)

    return scaler, X_train_scaled, X_val_scaled, X_test_scaled


def load_real_aviation_data(file_path: str = 'data/processed/consolidated_aviation_data.csv') -> pd.DataFrame:
    """
    Load real aviation data preprocessed by preprocess_real_data.py

    Parameters:
    -----------
    file_path : str
        Path to consolidated aviation data

    Returns:
    --------
    DataFrame : Consolidated real aviation emissions data
    """
    import os

    if os.path.exists(file_path):
        print(f"✓ Loading real aviation data from: {file_path}")
        df = pd.read_csv(file_path)
        print(f"✓ Loaded {len(df)} real flight records")
        return df
    else:
        print(f"⚠️  Real data not found at: {file_path}")
        print(f"   Run 'python preprocess_real_data.py' first, or use synthetic data.")
        return None


def prepare_data_pipeline(use_real_data: bool = True,
                          real_data_path: str = 'data/processed/consolidated_aviation_data.csv',
                          n_samples: int = 5000,
                          test_size: float = 0.15,
                          val_size: float = 0.15,
                          random_state: int = 42) -> Dict:
    """
    Complete data preparation pipeline.

    Parameters:
    -----------
    use_real_data : bool
        If True, load real aviation data; if False, generate synthetic data
    real_data_path : str
        Path to consolidated real aviation data
    n_samples : int
        Number of samples to generate (if using synthetic data)
    test_size : float
        Test set proportion
    val_size : float
        Validation set proportion
    random_state : int
        Random seed

    Returns:
    --------
    dict : Dictionary with all processed datasets and scaler
    """
    # Load or generate data
    if use_real_data:
        df = load_real_aviation_data(real_data_path)
        if df is None:
            print("   Falling back to synthetic data...")
            df = generate_synthetic_aviation_data(n_samples, random_state)
    else:
        df = generate_synthetic_aviation_data(n_samples, random_state)

    # Engineer features
    df = engineer_features(df)

    # Encode categorical features
    df = encode_categorical_features(df)

    # Separate features and target
    X = df.drop('co2_kg', axis=1)
    y = df['co2_kg']

    # Split data
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(
        X, y, test_size, val_size, random_state
    )

    # Scale features
    scaler, X_train_scaled, X_val_scaled, X_test_scaled = scale_features(
        X_train, X_val, X_test
    )

    return {
        'X_train': X_train,
        'X_val': X_val,
        'X_test': X_test,
        'y_train': y_train,
        'y_val': y_val,
        'y_test': y_test,
        'X_train_scaled': X_train_scaled,
        'X_val_scaled': X_val_scaled,
        'X_test_scaled': X_test_scaled,
        'scaler': scaler,
        'feature_names': X_train.columns.tolist()
    }


if __name__ == '__main__':
    print("Data processing module loaded successfully!")
    print("\nAvailable functions:")
    print("  - generate_synthetic_aviation_data()")
    print("  - engineer_features()")
    print("  - encode_categorical_features()")
    print("  - split_data()")
    print("  - scale_features()")
    print("  - prepare_data_pipeline()")
