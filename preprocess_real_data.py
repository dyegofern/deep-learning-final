"""
Real Aviation Dataset Preprocessing Pipeline
CSCA 5642 - Final Project

This script consolidates three real aviation datasets:
1. ICAO Aircraft Engine Emissions Databank (engine characteristics & emissions)
2. Aircraft Performance Dataset - Aircraft Bluebook (aircraft specifications & performance)
3. Aircraft Fuel Distribution System (fuel system sensor data & scenarios)

Run this script BEFORE running notebooks to create consolidated dataset.

Usage:
    python preprocess_real_data.py
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path
import glob
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# Configuration
# ============================================================================

RAW_DATA_DIR = Path('data/raw')
PROCESSED_DATA_DIR = Path('data/processed')

# Dataset paths
ICAO_ENGINE_PATH = RAW_DATA_DIR / 'ges.csv'  # ICAO gaseous emissions & smoke data
AIRCRAFT_PERFORMANCE_PATH = RAW_DATA_DIR / 'Airplane_Cleaned.csv'  # Aircraft bluebook data
FUEL_DISTRIBUTION_NORMAL = RAW_DATA_DIR / 'Scenario_Normal.csv'  # Normal fuel system operation
FUEL_DISTRIBUTION_SCENARIOS = [  # Abnormal scenarios
    RAW_DATA_DIR / 'Scenario_One.csv',
    RAW_DATA_DIR / 'Scenario_Two.csv',
    RAW_DATA_DIR / 'Scenario_Three.csv',
    RAW_DATA_DIR / 'Scenario_Four.csv'
]

# Output path
CONSOLIDATED_OUTPUT = PROCESSED_DATA_DIR / 'consolidated_aviation_data.csv'


# ============================================================================
# Dataset 1: ICAO Aircraft Engine Emissions Databank
# ============================================================================

def load_icao_engine_data(file_path: Path) -> pd.DataFrame:
    """
    Load and preprocess ICAO Aircraft Engine Emissions Databank.

    This dataset contains gaseous emissions and smoke data for production
    aircraft engines with thrust > 26.7 kN.

    Key columns:
    - Engine Identification
    - Rated Thrust (kN)
    - NOx EI T/O (g/kg) - Nitrogen Oxides at takeoff
    - CO EI T/O (g/kg) - Carbon Monoxide at takeoff
    - HC EI T/O (g/kg) - Hydrocarbons at takeoff
    - Fuel Flow T/O (kg/sec)
    - B/P Ratio - Bypass ratio
    - Pressure Ratio

    Returns:
    --------
    DataFrame with engine characteristics and emissions
    """
    print("=" * 70)
    print("LOADING ICAO ENGINE EMISSIONS DATA")
    print("=" * 70)

    if not file_path.exists():
        print(f"[WARN]  ICAO data not found at: {file_path}")
        print("   Creating sample engine mapping for demonstration...")
        return create_sample_engine_data()

    try:
        df = pd.read_csv(file_path)
        print(f"[OK] Loaded {len(df)} engine records")

        # Select relevant columns
        columns_of_interest = {
            'Engine Identification': 'engine_model',
            'Manufacturer': 'manufacturer',
            'Eng Type': 'engine_type',
            'Rated Thrust (kN)': 'thrust_kn',
            'B/P Ratio': 'bypass_ratio',
            'Pressure Ratio': 'pressure_ratio',
            'NOx EI T/O (g/kg)': 'nox_g_per_kg',
            'CO EI T/O (g/kg)': 'co_g_per_kg',
            'HC EI T/O (g/kg)': 'hc_g_per_kg',
            'Fuel Flow T/O (kg/sec)': 'fuel_flow_kg_s'
        }

        # Rename columns
        df = df.rename(columns=columns_of_interest)

        # Select only renamed columns that exist
        available_cols = [col for col in columns_of_interest.values() if col in df.columns]
        df = df[available_cols].copy()

        # Clean data - remove rows with missing critical values
        df = df.dropna(subset=['engine_model', 'thrust_kn', 'fuel_flow_kg_s'])

        # Fill missing emissions data with median values
        for col in ['nox_g_per_kg', 'co_g_per_kg', 'hc_g_per_kg']:
            if col in df.columns:
                df[col] = df[col].fillna(df[col].median())

        # Create aircraft type mapping based on engine model
        df['aircraft_type'] = df['engine_model'].apply(map_engine_to_aircraft)

        print(f"[OK] Processed {len(df)} engine records (after cleaning)")
        print(f"[OK] Columns: {list(df.columns)}")
        return df

    except Exception as e:
        print(f"[ERROR] Error loading ICAO data: {e}")
        import traceback
        traceback.print_exc()
        return create_sample_engine_data()


def map_engine_to_aircraft(engine_name: str) -> str:
    """Map engine model to typical aircraft type."""
    engine_name = str(engine_name).upper()

    # Common engine-to-aircraft mappings
    mappings = {
        'CFM56': 'A320',  # CFM56 used on A320, B737
        'CF6': 'B777',
        'GE90': 'B777',
        'TRENT': 'A380',
        'V2500': 'A320',
        'PW4000': 'B777',
        'LEAP': 'A320',
    }

    for key, aircraft in mappings.items():
        if key in engine_name:
            return aircraft

    return 'B737'  # Default


def create_sample_engine_data() -> pd.DataFrame:
    """Create sample engine data if ICAO file not available."""
    data = {
        'engine_model': ['CFM56-7B', 'V2500-A5', 'GE90-115B', 'TRENT-970', 'PW4000'],
        'aircraft_type': ['B737', 'A320', 'B777', 'A380', 'B777'],
        'thrust_kn': [120, 140, 510, 340, 400],
        'nox_g_per_kg': [15.2, 14.8, 18.5, 16.3, 17.1],
        'co_g_per_kg': [2.1, 2.3, 1.8, 1.9, 2.0],
        'fuel_flow_kg_s': [1.2, 1.3, 4.5, 3.8, 4.1]
    }
    print("  Using sample engine data (5 engines)")
    return pd.DataFrame(data)


# ============================================================================
# Dataset 2: Aircraft Performance Dataset (Aircraft Bluebook)
# ============================================================================

def load_aircraft_performance_data(file_path: Path) -> pd.DataFrame:
    """
    Load and preprocess Aircraft Performance Dataset (Bluebook).

    This dataset contains detailed specifications for 860+ aircraft including:
    - Physical dimensions (length, height, wingspan, weight)
    - Performance metrics (max speed, cruise speed, range, ceiling)
    - Engine specifications (type, thrust/horsepower)
    - Operational characteristics (takeoff/landing distances, climb rate)

    Key columns:
    - Model: Aircraft model name
    - Company: Manufacturer
    - Engine Type: Piston/Turboprop/Turbofan/Turbojet
    - Vmax: Maximum speed (knots)
    - Vcruise: Cruise speed (knots)
    - Range: Maximum range (nm)
    - FW: Fuel capacity (gallons)
    - AUW: All-up weight (lbs)
    - Hmax: Service ceiling (ft)

    Returns:
    --------
    DataFrame with aircraft performance characteristics
    """
    print("\n" + "=" * 70)
    print("LOADING AIRCRAFT PERFORMANCE DATA (BLUEBOOK)")
    print("=" * 70)

    if not file_path.exists():
        print(f"[WARN]  Aircraft performance data not found at: {file_path}")
        print("   Creating sample aircraft data for demonstration...")
        return create_sample_aircraft_data()

    try:
        df = pd.read_csv(file_path)
        print(f"[OK] Loaded {len(df)} aircraft records")

        # Standardize column names
        column_mapping = {
            'Model': 'aircraft_model',
            'Company': 'manufacturer',
            'Engine Type': 'engine_type',
            'THR': 'thrust_lbs',  # Thrust for jets
            'SHP': 'horsepower',  # Shaft horsepower for props
            'Vmax': 'max_speed_knots',
            'Vcruise': 'cruise_speed_knots',
            'Vstall': 'stall_speed_knots',
            'Range': 'range_nm',
            'FW': 'fuel_capacity_gal',
            'AUW': 'max_weight_lbs',
            'MEW': 'empty_weight_lbs',
            'Hmax': 'service_ceiling_ft',
            'ROC': 'rate_of_climb_fpm',
            'Length': 'length_ft',
            'Height': 'height_ft',
            'Wing Span': 'wingspan_ft'
        }

        df = df.rename(columns=column_mapping)

        # Select available columns
        available_cols = [col for col in column_mapping.values() if col in df.columns]
        df = df[available_cols].copy()

        # Clean data
        df = df.dropna(subset=['aircraft_model', 'manufacturer'])

        # Convert fuel capacity from gallons to kg (1 gal Jet-A ≈ 3.05 kg)
        if 'fuel_capacity_gal' in df.columns:
            df['fuel_capacity_kg'] = df['fuel_capacity_gal'] * 3.05

        # Convert weights from lbs to kg
        if 'max_weight_lbs' in df.columns:
            df['max_weight_kg'] = df['max_weight_lbs'] * 0.453592

        # Create simplified aircraft type categories (using both model and manufacturer)
        df['aircraft_type'] = df.apply(lambda row: categorize_aircraft_type(
            row.get('aircraft_model', ''),
            row.get('manufacturer', '')
        ), axis=1)

        print(f"[OK] Processed {len(df)} aircraft records")
        print(f"[OK] Columns: {list(df.columns)[:10]}... ({len(df.columns)} total)")
        return df

    except Exception as e:
        print(f"[ERROR] Error loading aircraft performance data: {e}")
        import traceback
        traceback.print_exc()
        return create_sample_aircraft_data()


def categorize_aircraft_type(model_name: str, manufacturer: str = '') -> str:
    """Categorize aircraft into simplified types based on model name and manufacturer."""
    model_upper = str(model_name).upper()
    mfr_upper = str(manufacturer).upper()

    # Commercial airliners (large jets)
    if any(x in model_upper for x in ['737', 'B737', 'BOEING 737']):
        return 'B737'
    elif any(x in model_upper for x in ['777', 'B777', 'BOEING 777']):
        return 'B777'
    elif any(x in model_upper for x in ['787', 'B787', 'BOEING 787', 'DREAMLINER']):
        return 'B787'
    elif any(x in model_upper for x in ['747', 'B747', 'BOEING 747']):
        return 'B747'
    elif any(x in model_upper for x in ['A320', 'AIRBUS 320']):
        return 'A320'
    elif any(x in model_upper for x in ['A321', 'AIRBUS 321']):
        return 'A321'
    elif any(x in model_upper for x in ['A380', 'AIRBUS 380']):
        return 'A380'
    elif any(x in model_upper for x in ['A350', 'AIRBUS 350']):
        return 'A350'
    elif any(x in model_upper for x in ['A330', 'AIRBUS 330']):
        return 'A330'

    # Business jets - Bombardier
    elif any(x in model_upper for x in ['CHALLENGER', 'CL-', 'CRJ']):
        return 'Bombardier_Challenger'
    elif any(x in model_upper for x in ['GLOBAL', 'BD-700']):
        return 'Bombardier_Global'
    elif any(x in model_upper for x in ['LEARJET', 'LEAR ']):
        return 'Learjet'

    # Business jets - Cessna
    elif any(x in model_upper for x in ['CITATION', 'CE-', 'C-500', 'C-550', 'C-560', 'C-650', 'C-680', 'C-700', 'C-750']):
        return 'Cessna_Citation'
    elif any(x in model_upper for x in ['CARAVAN', 'GRAND CARAVAN']):
        return 'Cessna_Caravan'
    elif 'CESSNA' in model_upper and 'JET' in model_upper:
        return 'Cessna_Jet'
    elif 'CESSNA' in model_upper:
        return 'Cessna_Turboprop'

    # Business jets - Gulfstream
    elif any(x in model_upper for x in ['GULFSTREAM', 'G-', 'GII', 'GIII', 'GIV', 'GV', 'G100', 'G200', 'G300', 'G400', 'G450', 'G500', 'G550', 'G600', 'G650', 'G700', 'G800']):
        return 'Gulfstream'

    # Business jets - Dassault
    elif any(x in model_upper for x in ['FALCON', 'MYSTERE']):
        return 'Dassault_Falcon'

    # Business jets - Embraer
    elif any(x in model_upper for x in ['PHENOM', 'PRAETOR', 'LEGACY', 'LINEAGE']):
        return 'Embraer_Business'
    elif any(x in model_upper for x in ['ERJ', 'E-JET', 'E170', 'E175', 'E190', 'E195']):
        return 'Embraer_Regional'

    # Business jets - Beechcraft/Hawker
    elif any(x in model_upper for x in ['KING AIR', 'SUPER KING AIR', 'B200', 'C90', 'F90', 'E90', 'B100', 'A100']):
        return 'Beechcraft_KingAir'
    elif any(x in model_upper for x in ['BEECHJET', 'HAWKER', 'PREMIER']):
        return 'Beechcraft_Jet'
    elif 'STARSHIP' in model_upper:
        return 'Beechcraft_Starship'

    # Turboprops - General
    elif any(x in model_upper for x in ['PILATUS', 'PC-']):
        return 'Pilatus'
    elif any(x in model_upper for x in ['PIPER']):
        return 'Piper_Turboprop'
    elif any(x in model_upper for x in ['COMMANDER', 'TWIN COMMANDER']):
        return 'Twin_Commander'

    # Agricultural aircraft
    elif any(x in model_upper for x in ['AIR TRACTOR', 'AT-']):
        return 'Air_Tractor'
    elif any(x in model_upper for x in ['AG CAT', 'AGCAT']):
        return 'Ag_Cat'

    # Military/Special
    elif any(x in model_upper for x in ['SABRELINER', 'T-39']):
        return 'Sabreliner'
    elif any(x in model_upper for x in ['JETPROP', 'ROCKET']):
        return 'Modified_Turboprop'

    # Check manufacturer for additional clues
    elif 'BOMBARDIER' in mfr_upper or 'CANADAIR' in mfr_upper:
        return 'Bombardier_Other'
    elif 'MITSUBISHI' in mfr_upper or 'DIAMOND' in mfr_upper:
        return 'Mitsubishi_Diamond'
    elif 'FAIRCHILD' in mfr_upper or 'M7' in mfr_upper or 'DORNIER' in mfr_upper:
        return 'Fairchild_Dornier'
    elif 'IAI' in mfr_upper or 'ISRAEL' in mfr_upper or 'ASTRA' in mfr_upper or 'WESTWIND' in mfr_upper:
        return 'IAI'
    elif 'SOCATA' in mfr_upper or 'TBM' in model_upper:
        return 'Socata_TBM'
    elif 'ECLIPSE' in mfr_upper or 'ECLIPSE' in model_upper:
        return 'Eclipse'
    elif 'HONDA' in mfr_upper or 'HONDAJET' in model_upper:
        return 'HondaJet'
    elif 'CIRRUS' in mfr_upper or 'VISION' in model_upper and 'JET' in model_upper:
        return 'Cirrus_VisionJet'
    elif 'LOCKHEED' in mfr_upper or 'JETSTAR' in model_upper:
        return 'Lockheed'
    elif 'DE HAVILLAND' in mfr_upper or 'DASH' in model_upper:
        return 'DeHavilland'
    elif 'AYRES' in mfr_upper:
        return 'Ayres_Turbine'

    # Default categories
    else:
        return 'Other_Turbine'


def create_sample_aircraft_data() -> pd.DataFrame:
    """Create sample aircraft data if actual file not available."""
    np.random.seed(42)

    aircraft_types = ['B737', 'A320', 'B777', 'B787', 'A380']
    data = []

    for i, ac_type in enumerate(aircraft_types):
        # Generate specs based on aircraft type
        if ac_type in ['B737', 'A320']:  # Narrow-body
            max_weight = np.random.uniform(70000, 90000)
            fuel_capacity = np.random.uniform(20000, 26000)
            cruise_speed = np.random.uniform(450, 490)
            range_nm = np.random.uniform(3000, 3500)
        elif ac_type in ['B777', 'B787']:  # Wide-body
            max_weight = np.random.uniform(250000, 350000)
            fuel_capacity = np.random.uniform(90000, 140000)
            cruise_speed = np.random.uniform(480, 520)
            range_nm = np.random.uniform(7000, 9000)
        else:  # A380
            max_weight = np.random.uniform(500000, 575000)
            fuel_capacity = np.random.uniform(250000, 320000)
            cruise_speed = np.random.uniform(490, 510)
            range_nm = np.random.uniform(8000, 8500)

        data.append({
            'aircraft_model': f'{ac_type}-{i+1}00',
            'manufacturer': 'Boeing' if 'B' in ac_type else 'Airbus',
            'engine_type': 'Turbofan',
            'max_speed_knots': cruise_speed + np.random.uniform(10, 30),
            'cruise_speed_knots': cruise_speed,
            'range_nm': range_nm,
            'fuel_capacity_kg': fuel_capacity,
            'max_weight_kg': max_weight,
            'service_ceiling_ft': np.random.uniform(39000, 43000),
            'aircraft_type': ac_type
        })

    print(f"  Using sample aircraft data ({len(data)} aircraft)")
    return pd.DataFrame(data)


# ============================================================================
# Dataset 3: Aircraft Fuel Distribution System
# ============================================================================

def load_fuel_distribution_data(normal_path: Path, scenario_paths: List[Path]) -> pd.DataFrame:
    """
    Load aircraft fuel distribution system sensor data.

    This dataset tracks fuel system sensors across normal and abnormal scenarios.
    Sensor measurements include:
    - FTL: Fuel Tank Level
    - CTL: Center Tank Level
    - FTF: Fuel Tank Flow
    - FTV_S: Fuel Tank Valve Status
    - CLF: Center Line Flow
    - CLV_S: Center Line Valve Status
    - FTT: Fuel Tank Temperature
    - CRTT: Center Tank Temperature

    Five scenarios:
    - Scenario_Normal: Normal operations
    - Scenario_One through Four: Various abnormal conditions

    Returns:
    --------
    DataFrame with fuel system sensor data and scenario labels
    """
    print("\n" + "=" * 70)
    print("LOADING FUEL DISTRIBUTION SYSTEM DATA")
    print("=" * 70)

    all_scenarios = []

    # Load normal scenario
    if normal_path.exists():
        try:
            df_normal = pd.read_csv(normal_path)
            df_normal['scenario'] = 'Normal'
            df_normal['is_abnormal'] = 0
            all_scenarios.append(df_normal)
            print(f"[OK] Loaded Normal scenario: {len(df_normal)} records")
        except Exception as e:
            print(f"[WARN]  Error loading normal scenario: {e}")

    # Load abnormal scenarios
    for i, scenario_path in enumerate(scenario_paths, 1):
        if scenario_path.exists():
            try:
                df_scenario = pd.read_csv(scenario_path)
                df_scenario['scenario'] = f'Abnormal_{i}'
                df_scenario['is_abnormal'] = 1
                all_scenarios.append(df_scenario)
                print(f"[OK] Loaded Abnormal scenario {i}: {len(df_scenario)} records")
            except Exception as e:
                print(f"[WARN]  Error loading abnormal scenario {i}: {e}")

    if len(all_scenarios) == 0:
        print("[WARN]  No fuel distribution data found, creating sample data...")
        return create_sample_fuel_distribution_data()

    # Consolidate all scenarios
    df = pd.concat(all_scenarios, ignore_index=True)

    print(f"[OK] Consolidated {len(all_scenarios)} scenarios ({len(df)} total records)")
    print(f"[OK] Columns: {list(df.columns)}")

    return df


def create_sample_fuel_distribution_data() -> pd.DataFrame:
    """Create sample fuel distribution data if actual files not available."""
    np.random.seed(42)

    data = []
    scenarios = ['Normal'] + [f'Abnormal_{i}' for i in range(1, 5)]

    for scenario in scenarios:
        is_abnormal = 0 if scenario == 'Normal' else 1
        n_records = 50

        for _ in range(n_records):
            # Normal vs abnormal patterns
            if is_abnormal:
                ftl = np.random.uniform(50, 100)  # More variation
                ctl = np.random.uniform(50, 100)
                ftf = np.random.uniform(3, 7)
                ftt = np.random.uniform(-25, -15)  # Temperature issues
            else:
                ftl = np.random.uniform(95, 105)  # Stable
                ctl = np.random.uniform(95, 105)
                ftf = np.random.uniform(4.5, 5.5)
                ftt = np.random.uniform(-22, -18)

            data.append({
                'FTL': ftl,
                'CTL': ctl,
                'FTF': ftf,
                'FTV_S': np.random.uniform(4, 6),
                'CLF': np.random.uniform(-0.5, 0.5),
                'CLV_S': np.random.uniform(-0.5, 0.5),
                'FTT': ftt,
                'CRTT': ftt + np.random.uniform(-2, 2),
                'scenario': scenario,
                'is_abnormal': is_abnormal
            })

    df = pd.DataFrame(data)
    print(f"  Using sample fuel distribution data ({len(df)} records)")
    return df


# ============================================================================
# Consolidation Logic
# ============================================================================

def consolidate_datasets(icao_df: pd.DataFrame,
                        aircraft_df: pd.DataFrame,
                        fuel_dist_df: pd.DataFrame) -> pd.DataFrame:
    """
    Consolidate all three datasets into unified aviation emissions dataset.

    Strategy:
    1. Use aircraft performance data as base (provides aircraft specs)
    2. Merge with ICAO engine data (adds engine characteristics & emissions)
    3. Add fuel distribution system data for operational parameters
    4. Engineer features and calculate CO2 emissions

    Returns:
    --------
    Consolidated DataFrame ready for ML pipeline with same output format
    """
    print("\n" + "=" * 70)
    print("CONSOLIDATING DATASETS")
    print("=" * 70)

    # Filter aircraft data to jet aircraft only (for aviation emissions modeling)
    # Engine types in this dataset: 'Jet', 'Propjet', 'Piston'
    aircraft_df_jets = aircraft_df[
        aircraft_df['engine_type'].isin(['Jet', 'Propjet'])
    ].copy()

    print(f"Base dataset: {len(aircraft_df_jets)} jet/turboprop aircraft from performance data")

    # Create synthetic flight records from aircraft data
    # Each aircraft gets multiple flight scenarios
    np.random.seed(42)
    flight_records = []

    phases = ['taxi', 'takeoff', 'climb', 'cruise', 'descent', 'approach', 'landing']

    for _, aircraft in aircraft_df_jets.iterrows():
        # Generate 3-5 flight records per aircraft
        n_flights = np.random.randint(3, 6)

        for flight_num in range(n_flights):
            phase = np.random.choice(phases)

            # Phase-specific parameters
            if phase == 'taxi':
                altitude = np.random.uniform(0, 50)
                speed = aircraft.get('cruise_speed_knots', 450) * np.random.uniform(0.02, 0.05)
            elif phase in ['takeoff', 'climb']:
                altitude = np.random.uniform(1000, 20000)
                speed = aircraft.get('cruise_speed_knots', 450) * np.random.uniform(0.5, 0.8)
            elif phase == 'cruise':
                altitude = aircraft.get('service_ceiling_ft', 35000) * np.random.uniform(0.8, 0.95)
                speed = aircraft.get('cruise_speed_knots', 450) * np.random.uniform(0.95, 1.0)
            elif phase in ['descent', 'approach']:
                altitude = np.random.uniform(500, 15000)
                speed = aircraft.get('cruise_speed_knots', 450) * np.random.uniform(0.4, 0.7)
            else:  # landing
                altitude = np.random.uniform(0, 200)
                speed = aircraft.get('cruise_speed_knots', 450) * np.random.uniform(0.25, 0.35)

            # Calculate route distance (varies by phase)
            if phase == 'cruise':
                route_distance = aircraft.get('range_nm', 3000) * np.random.uniform(0.3, 0.9)
            else:
                route_distance = np.random.uniform(50, 500)

            flight_records.append({
                'aircraft_type': aircraft.get('aircraft_type', 'Other'),
                'aircraft_model': aircraft.get('aircraft_model', 'Unknown'),
                'manufacturer': aircraft.get('manufacturer', 'Unknown'),
                'flight_phase': phase,
                'altitude_ft': altitude,
                'speed_knots': speed,
                'weight_tons': aircraft.get('max_weight_kg', 70000) / 1000,  # Convert to tons
                'route_distance_nm': route_distance,
                'fuel_capacity_kg': aircraft.get('fuel_capacity_kg', 20000),
                'max_speed_knots': aircraft.get('max_speed_knots', speed),
                'cruise_speed_knots': aircraft.get('cruise_speed_knots', 450)
            })

    consolidated = pd.DataFrame(flight_records)
    print(f"Generated {len(consolidated)} flight records from aircraft data")

    # Merge with ICAO engine data (by aircraft type)
    # Calculate average engine characteristics per aircraft type
    icao_aggregated = icao_df.groupby('aircraft_type').agg({
        'thrust_kn': 'mean',
        'nox_g_per_kg': 'mean',
        'co_g_per_kg': 'mean',
        'fuel_flow_kg_s': 'mean'
    }).reset_index()

    consolidated = consolidated.merge(
        icao_aggregated,
        on='aircraft_type',
        how='left'
    )

    # Fill missing engine data with type-based estimates
    consolidated['thrust_kn'] = consolidated['thrust_kn'].fillna(
        consolidated['weight_tons'] * 0.3  # Rough thrust-to-weight estimate
    )
    consolidated['fuel_flow_kg_s'] = consolidated['fuel_flow_kg_s'].fillna(
        consolidated['thrust_kn'] * 0.01  # Rough fuel flow estimate
    )
    consolidated['nox_g_per_kg'] = consolidated['nox_g_per_kg'].fillna(16.0)
    consolidated['co_g_per_kg'] = consolidated['co_g_per_kg'].fillna(2.0)

    print(f"After ICAO merge: {len(consolidated)} records")

    # Add environmental features
    consolidated['temperature_c'] = np.random.uniform(-55, 30, len(consolidated))
    consolidated['wind_speed_knots'] = np.random.uniform(-50, 50, len(consolidated))

    # Add fuel distribution system features (sample from normal operations)
    if len(fuel_dist_df) > 0:
        normal_fuel_data = fuel_dist_df[fuel_dist_df['scenario'] == 'Normal']
        if len(normal_fuel_data) > 0:
            # Sample fuel system metrics
            fuel_samples = normal_fuel_data.sample(len(consolidated), replace=True).reset_index(drop=True)
            consolidated['fuel_tank_level'] = fuel_samples['FTL'].values
            consolidated['fuel_flow_rate'] = fuel_samples['FTF'].values
            consolidated['fuel_temp_c'] = fuel_samples['FTT'].values
        else:
            consolidated['fuel_tank_level'] = np.random.uniform(80, 100, len(consolidated))
            consolidated['fuel_flow_rate'] = np.random.uniform(4, 6, len(consolidated))
            consolidated['fuel_temp_c'] = np.random.uniform(-25, -15, len(consolidated))
    else:
        consolidated['fuel_tank_level'] = np.random.uniform(80, 100, len(consolidated))
        consolidated['fuel_flow_rate'] = np.random.uniform(4, 6, len(consolidated))
        consolidated['fuel_temp_c'] = np.random.uniform(-25, -15, len(consolidated))

    # Calculate fuel consumption and CO2 emissions (target variable)
    # Estimate based on distance, weight, and fuel flow
    consolidated['fuel_consumed_kg'] = (
        consolidated['route_distance_nm'] *
        consolidated['weight_tons'] *
        0.05 *  # Fuel burn factor
        (consolidated['speed_knots'] / consolidated['cruise_speed_knots'])
    )

    # CO2 emissions: 3.16 kg CO2 per kg of Jet-A fuel burned
    consolidated['co2_kg'] = consolidated['fuel_consumed_kg'] * 3.16

    # Select final columns (maintaining same output format as before)
    final_columns = [
        'aircraft_type',
        'flight_phase',
        'altitude_ft',
        'speed_knots',
        'weight_tons',
        'route_distance_nm',
        'temperature_c',
        'wind_speed_knots',
        'thrust_kn',
        'fuel_flow_kg_s',
        'co2_kg'  # Target variable
    ]

    consolidated = consolidated[final_columns].dropna()

    print(f"\n[OK] Final consolidated dataset: {len(consolidated)} records")
    print(f"[OK] Features: {len(final_columns)}")
    print(f"[OK] Target: co2_kg (CO2 emissions)")

    return consolidated


# ============================================================================
# Main Preprocessing Pipeline
# ============================================================================

def main():
    """Main preprocessing pipeline."""
    print("\n" + "=" * 70)
    print("AVIATION EMISSIONS DATA PREPROCESSING PIPELINE")
    print("CSCA 5642 - Final Project")
    print("=" * 70)
    print("Datasets:")
    print("  1. ICAO Aircraft Engine Emissions Databank")
    print("  2. Aircraft Performance Dataset (Aircraft Bluebook)")
    print("  3. Aircraft Fuel Distribution System")
    print("=" * 70)

    # Create output directory
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Load individual datasets
    icao_df = load_icao_engine_data(ICAO_ENGINE_PATH)
    aircraft_df = load_aircraft_performance_data(AIRCRAFT_PERFORMANCE_PATH)
    fuel_dist_df = load_fuel_distribution_data(FUEL_DISTRIBUTION_NORMAL, FUEL_DISTRIBUTION_SCENARIOS)

    # Consolidate
    consolidated = consolidate_datasets(icao_df, aircraft_df, fuel_dist_df)

    # Save consolidated dataset
    consolidated.to_csv(CONSOLIDATED_OUTPUT, index=False)

    print("\n" + "=" * 70)
    print("PREPROCESSING COMPLETE")
    print("=" * 70)
    print(f"[OK] Output saved to: {CONSOLIDATED_OUTPUT}")
    print(f"[OK] Total records: {len(consolidated)}")
    print(f"\nDataset Preview:")
    print(consolidated.head(10))
    print(f"\nDataset Statistics:")
    print(consolidated.describe())

    print("\n" + "=" * 70)
    print("NEXT STEPS")
    print("=" * 70)
    print("1. Review the consolidated dataset in: data/processed/")
    print("2. Run notebooks/01_data_preparation.ipynb")
    print("3. The notebook will load this consolidated data")
    print("=" * 70)


if __name__ == '__main__':
    main()
