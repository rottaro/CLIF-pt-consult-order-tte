import pandas as pd
import pytz
from datetime import datetime
from datetime import timedelta
import os
import json
import clifpy

with open('../config/config.json', 'r') as file:
    config = json.load(file)

#To use MIMIC which we should not need once all data is in CLIF format
use_mimic = config["use_mimic"]

#file paths
path_out = config['output']

def ensure_datetime_est(s: pd.Series) -> pd.Series:
    """Parse to datetime; convert UTC into EST"""
    s = pd.to_datetime(s, errors='coerce')
    try:
        # Convert to EST if timezone-aware
        if s.dt.tz is not None:
            return s.dt.tz_convert('America/New_York')
        else:
            return s.dt.tz_localize('America/New_York')
    except (TypeError, AttributeError):
        return s

def convert_datetime_columns_to_eastern(df: pd.DataFrame) -> pd.DataFrame:
    """Applies ensure_datetime_est to all datetime columns."""
    df = df.copy()
    for col in df.columns:
        # Only operate on datetime-like columns
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            df[col] = ensure_datetime_est(df[col])
        
    return df

#Load ClifOrchestrator
co = clifpy.ClifOrchestrator(
    data_directory=config['clif'],
    filetype='parquet',
    timezone='America/New_York',
    output_directory=config['output']
)
def load_clif_table(table_name:str, hosp_list:list = []) -> pd.DataFrame:
    '''
    hosp_list is a list of hospitalization_ids to filter by.
    '''
    if hosp_list:
        co.initialize(tables = [table_name],
                      filters={'hospitalization_id':hosp_list} )
    else:
        co.initialize(tables = [table_name])
    table = co.load_table(table_name)
    clifpy.utils.apply_outlier_handling(table) #Remove outliers per CLIF standards
    return table.df

def load_data(data_set:str, name:str, folder:str = '', type:str = 'parquet') -> pd.DataFrame:
    """Loads a table from the MIMIC/mobilization/output data sets. Input: string with table name. Output: DataFrame. Automatically converts to EST."""
    if folder:
        path = os.path.join(config[data_set], folder, f"{name}.{type}")
    else:
        path = os.path.join(config[data_set], f"{name}.{type}")
    
    if type == 'parquet':
        df = pd.read_parquet(path)
    elif type == 'csv' or type == 'csv.gz':
        df = pd.read_csv(path)
    else:
        raise ValueError('File type needs to be Parquet, csv or cvs.gz.')
    df = convert_datetime_columns_to_eastern(df)
    return df

def aggregate_by_time(df:pd.DataFrame, val_col:str, min_time:int, max_time:int, agg_func:str = 'mean'):
    '''
    INPUTS:
    df = DataFrame required to have:
        column called 'time_diff' which is a date time difference value or any number (assumed hours)
        val_col
        encounter_block
    min_time, max_time = hours as an int, +/-999 representing infinity.
    agg = aggregation function to pivot table
    '''
    df = df[['encounter_block','time_diff',val_col]].copy()
    #If time_delta format convert to hours.
    if pd.api.types.is_timedelta64_dtype(df['time_diff']):
        df['time_diff'] = df['time_diff'].dt.total_seconds()/3600
    
    #Time mask, note that -999 and +999 are treated as infinate
    if min_time == -999:
        time_mask = df['time_diff'] < max_time
    elif max_time == 999:
        time_mask = df['time_diff'] > min_time
    else:
        time_mask = df['time_diff'].between(min_time, max_time)
    df = df[time_mask]
    
    #Aggregation, rename the variable for usefulness
    new_name = f"{val_col}_{min_time}_{max_time}h_{agg_func}"
    agg_df = df.groupby('encounter_block')[val_col].agg(agg_func).reset_index()
    agg_df.rename(columns={val_col:new_name},inplace=True)
    return agg_df
    
#Used to break variables into time windows
def classify_time_window(td):
    if type(td) is int:
        hours = td
    else:
        hours =  td.total_seconds() / 3600
    
    if 0 <= hours < 24:
        return '0-24h'
    elif 24 <= hours < 48:
        return '24-48h'
    elif 48 <= hours < 72:
        return '48-72h'
    else:
        return None

#Used to break variables into time windows
def classify_time_window_ext(td):
    if type(td) is int:
        hours = td
    else:
        hours =  td.total_seconds() / 3600
    
    if hours < -24:
        return 'past'
    if -24 <= hours < 0:
        return 'prior24'
    elif 0 <= hours < 24:
        return '0-24h'
    elif 24 <= hours < 48:
        return '24-48h'
    elif 48 <= hours < 72:
        return '48-72h'
    elif hours > 72:
        return 'future'
    else:
        return None