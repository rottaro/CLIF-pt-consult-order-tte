import pandas as pd
import pytz
from datetime import datetime
from datetime import timedelta
import os
import json
import clifpy
import numpy as np

with open('../config/config.json', 'r') as file:
    config = json.load(file)

#To use MIMIC which we should not need once all data is in CLIF format
use_mimic = config["use_mimic"]

#file paths
path_out = config['output']

#My time_zone
my_tz = config['time_zone']

#Time Zone things
def ensure_datetime(s: pd.Series) -> pd.Series:
    """Parse to datetime; convert UTC into my_tz"""
    s = pd.to_datetime(s, errors='coerce')
    try:
        # Convert to EST if timezone-aware
        if s.dt.tz is not None:
            return s.dt.tz_convert(my_tz)
        else:
            return s.dt.tz_localize(my_tz)
    except (TypeError, AttributeError):
        return s

def convert_datetime_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Applies ensure_datetime_est to all datetime columns."""
    df = df.copy()
    for col in df.columns:
        # Only operate on datetime-like columns
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            df[col] = ensure_datetime(df[col])
        
    return df

#Load ClifOrchestrator
co = clifpy.ClifOrchestrator(
    data_directory=config['clif'],
    filetype='parquet',
    timezone=my_tz,
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
    """Loads a table from the MIMIC/mobilization/output data sets. Input: string with table name. Output: DataFrame. Automatically converts to my_tz."""
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
    df = convert_datetime_columns(df)
    return df

#time_bins object
class time_bins:
    
    def __init__(self, eb:pd.DataFrame):
        '''
        INPUT
        DataFrame with UNIQUE encounter_blocks per row, you can also have other additional columns in there without issue.
        '''
        self.bin_array = np.arange(-1*config['time_bin_size'], config['time_end'] + config['time_bin_size'], config['time_bin_size'])
        self.bins_df = pd.DataFrame({
            'bin_start': self.bin_array[:-1],
            'bin_end': self.bin_array[1:]
        })
        self.bins_df['time_bin'] = self.bins_df['bin_start'].astype(str)
        self.df = pd.merge(eb, self.bins_df, how='cross').reset_index()
        self.df = self.df.sort_values(by=['encounter_block','bin_start'])
    
    def classify_time_bin(self, t_diff:pd.Series) -> pd.Series:
        '''
        INPUT
        t_diff = Pandas DataSeries with a time_delta time element or a numerical element assumed to be hours
        OUTPUT
        DataSeries with time_bins[time_bin] as the value.
        '''
        #If time_delta format convert to hours.
        if pd.api.types.is_timedelta64_dtype(t_diff):
            t_diff = t_diff.dt.total_seconds()/3600
    
        return pd.cut(t_diff, bins=self.bin_array, labels=self.bins_df['time_bin'].tolist(), right=True, include_lowest=True)

    def bin_sort_fill(self,
                      val_col:str,
                      fill_with):
        '''
        Sorts time_bins and fill them in with specified request.
        INPUT
        val_col = val_col string name in the self.df.
        fill_with = either forward fill or 1
        '''
        
        if fill_with == 'ffill':
            self.df = self.df.sort_values(by=['encounter_block','bin_start']) #A bit repetitive but hard to have do this other way
            self.df[val_col] = self.df.groupby('encounter_block')[val_col].ffill()
        else:
            self.df[val_col].fillna(fill_with, inplace=True)
    
    def gather_time_bins(self,
                         input_df:pd.DataFrame,
                         val_col:str,
                         agg_func:str = 'mean',
                         fill_with = np.nan):
        '''
        INPUTS:
        df = DataFrame required to have:
            Either a column called 'time_diff' which is a date time difference value or any number (assumed hours) or 'time_bin'.
            val_col, the data itself to be aggregated.
            encounter_block
            time_bin = unique time string can be created with function above.
        agg = aggregation function to aggregate
        '''
        #Time bins
        if 'time_bin' not in input_df.columns:
            input_df['time_bin'] = self.classify_time_bin(input_df['time_diff'])
    
        #Copy data frame
        in_df = input_df[['encounter_block','time_bin',val_col]].copy()
        #Remove time windows outside of the time bins
        in_df = in_df[in_df['time_bin'].notna()]
        
        #Name of new variable
        new_name = f"{val_col}_{agg_func}"
    
        #Flag for agg_func = flag or all
        or_flag = agg_func == 'flag'
        if or_flag:
            agg_func = 'max'
            in_df[val_col] = in_df[val_col].astype(bool)
        and_flag = agg_func == 'all'
        if and_flag:
            agg_func = 'mean'
            in_df[val_col] = in_df[val_col].astype(bool)
    
        #Aggregation step
        print(f'Aggregation step for {new_name}')
        agg_df = in_df.groupby(by=['encounter_block','time_bin'])[val_col].agg(agg_func).reset_index()

        #Rename column
        agg_df.rename(columns={val_col:new_name},inplace=True)
        
        #Spcial situation for "or" and "and" to convert to boolean and back to int to 0 or 1.
        if or_flag or and_flag:
            agg_df[new_name] = agg_df[new_name] == 1
            agg_df[new_name] = agg_df[new_name].astype(int)
        
        #Merge onto time bins
        print(f'Merging step for {new_name}')
        self.df = self.df.merge(agg_df, on=['encounter_block','time_bin'], how='left', sort=False)

        #Sort and fill
        print(f'Sort/Fill step for {new_name}')
        self.bin_sort_fill(new_name, fill_with)

    def add_event(self,
                  input_df:pd.DataFrame,
                  new_name:str):
        '''
        INPUT
        in_df: A data frame which requires 'encounter_block', and either 'time_diff' or 'time_bin'.
        new_name: a string with a name for the new columns.
        NOTE: It will be back fill with 0 and forward fill with 1.
        '''
        #Time bins
        if ~('time_bin' in input_df.columns):
            input_df['time_bin'] = self.classify_time_bin(input_df['time_diff'])
    
        #Copy data frame
        in_df = input_df[['encounter_block','time_bin']].copy().drop_duplicates()
        #Remove time windows outside of the time bins
        in_df = in_df[in_df['time_bin'].notna()]
        #Dummy true variable
        in_df[new_name] = 1
        #Put into bin architecture
        self.df = pd.merge(self.df, in_df, on=['encounter_block','time_bin'], how='left', sort=False)
        #Fill ahead with 1 if present.
        self.bin_sort_fill(new_name, 'ffill')
        #Fill everything else with zero
        self.bin_sort_fill(new_name, 0)
        in_df[new_name] = in_df[new_name].astype(bool)

    def remove_based_on_censor(self, censor_col:str, keep_first:bool = False):
        '''
        Removes rows from the time bins based on some censoring variable.
        INPUT
        censor_col: String with the name of the column in time bins to use for this.
        keep_first: Boolean to either keep the first columns to be censored (ie for death where we keep the bin in which they died but no the subsequent bins).
        '''
        start_N = self.df.shape[0]
        if keep_first:
            rm = self.df.duplicated(subset=['encounter_block',censor_col], keep='first') & self.df[censor_col]
        else:
            rm = self.df[censor_col]
        self.df = self.df[~rm]
        end_N = self.df.shape[0]
        print(f"Removed {start_N - end_N} out of {start_N} from time_bin.df based on censor {censor_col} with keep_first = {keep_first}")
    
    def table_summary(self) -> pd.DataFrame:
        def missing(x): return round(np.mean(x.isna())*100,2)
        numeric_df = self.df.select_dtypes(include=['number','bool'])
        numeric_df['time_bin'] = self.df['time_bin']
        return  numeric_df.groupby('time_bin').agg(['min','mean','max', missing]).reset_index()

    def save(self):
        path = os.path.join(path_out, "time_bins.csv")
        self.df.to_csv(path)
        path = os.path.join(path_out, "time_bins_summary.csv")
        self.table_summary().to_csv(path)
        del path

#Older aggregation by time function replaced by time bins above
def aggregate_by_time(in_df:pd.DataFrame, val_col:str, min_time:int = 0, max_time:int = 24, agg_func:str = 'mean'):
    '''
    INPUTS:
    df = DataFrame required to have:
        column called 'time_diff' which is a date time difference value or any number (assumed hours)
        val_col
        encounter_block
    min_time, max_time = hours as an int, +/-999 representing infinity.
    agg = aggregation function to pivot table
    '''
    df = in_df[['encounter_block','time_diff',val_col]].copy()
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
    
    #Rename the variable for usefulness
    new_name = f"{val_col}_{min_time}_{max_time}h_{agg_func}"
    
    #Flag for agg_func = flag or all
    or_flag = agg_func == 'flag'
    if or_flag:
        agg_func = 'max'
        df[val_col] = df[val_col].astype(bool)
    and_flag = agg_func == 'all'
    if and_flag:
        agg_func = 'mean'
        df[val_col] = df[val_col].astype(bool)

    print(f'Aggregation step for {new_name}')
    agg_df = df.groupby('encounter_block')[val_col].agg(agg_func).reset_index()
    agg_df.rename(columns={val_col:new_name},inplace=True)

    #Spcial situation for "true" and "all" to convert to boolean
    if or_flag or and_flag: agg_df[new_name] = agg_df[new_name] == 1
    
    return agg_df
    