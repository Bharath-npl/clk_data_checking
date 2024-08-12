
# *********************************************
# ****************** Clock Data Checking ******


# to tune this code in your local PC use the follwoing command **
# streamlit run .\clk_data_checking.py --server.port 8888

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta, date, time
import numpy as np
import io
from streamlit_option_menu import option_menu
import datetime
from io import StringIO
import plotly.express as px
from dateutil.parser import parse
from datetime import datetime
import matplotlib.pyplot as plt
from PIL import Image
import json
from streamlit_plotly_events import plotly_events
import math
import scipy.stats  # used in confidence_intervals()



st.set_page_config(page_title="Clock Data Checking", page_icon=":stopwatch:", layout="wide")
st.markdown('<style>div.block-container{padding-top:1rem;}</style>', unsafe_allow_html=True)
st.title(":clock10: Clock Data Checking")



def validate_timestamp_column(column):
    # st.write(f"column: {column}")
    if column.empty:
        st.write("Time stamp column is EMPTY")
        return False
    
    sample_value = column.dropna().iloc[0]  # Get the first non-null value for testing
    # st.write(f"sample value: {sample_value}")
    # st.write(type(sample_value))
    # Various expected datetime formats common in our data as time stamps
    datetime_formats = [
        '%Y-%m-%d %H:%M:%S',  # Year-month-day Hour:Minute:Second
        '%H:%M:%S',  # Hour:Minute:Second only
        '%Y-%m-%d',  # Year-month-day only
        '%d-%m-%Y'   # Date-month-Year
    ]

    # Attempt to handle MJD or numeric timestamp formats first
    if isinstance(sample_value, (int, float, np.int64, pd.Timestamp)):
        # st.write("Its a float or int data")
        try:
            numeric_value = float(sample_value)
            # Assuming the sample value is in MJD format or similar
            if 40000 <= numeric_value <= 90000:  # Typical MJD range check
                
                return True
        except ValueError:
            # The value is not a float, so it could be a string representation of a date
            pass

        # Fallback: Try parsing as a string date if not a recognizable MJD
        # try:
        #     parse(sample_value)
        #     return True  # Parsing succeeded, it's a valid date string
        # except (ValueError, TypeError):
        #     return False  # Parsing failed, not a valid date string
    
    if isinstance(sample_value, str):
        # Remove commas from the string
        # st.write("Its a string data")
        sample_value_cleaned = sample_value.replace(',', '')
        st.write(sample_value_cleaned)
        # Check for MJD string with decimal values
        try:
            numeric_value = float(sample_value_cleaned)
            # st.write(f"Numeric value cleaned: {numeric_value}")
            if 40000 <= numeric_value <= 90000:  # Typical MJD range check
                return True
        except ValueError:
            pass  # Continue to next checks

        # Check for standard datetime formats
        for fmt in datetime_formats:
            try:
                datetime.strptime(sample_value_cleaned, fmt)
                return True  # Valid datetime format found
            except ValueError:
                continue  # Try next format
    st.write("No valid format found")
    return False  # No valid format found


def validate_value_column(column):
    try:
        pd.to_numeric(column.dropna(), errors='raise')  # Attempt to convert the entire column to numeric types
        return True
    except (ValueError, TypeError):
        return False

# Define a class to represent each column
class DataColumn:
    def __init__(self, index, column_type):
        self.index = index
        self.column_type = column_type  # 'timestamp' or 'value'

# Create DataColumn objects for each column index
timestamp_column = DataColumn(0, 'timestamp')
value_column = DataColumn(1, 'value')


#     return True  # All selected columns are valid

def validate_column_selections(df):
    # Since df only contains the two user-selected columns, we know their indices are 0 and 1
    # We can validate directly without needing user provided indices.

    # Validate timestamp column (assumed to always be the first column in df)
    if not validate_timestamp_column(df.iloc[:, 0]):
        st.error("The selected timestamp column does not contain valid timestamp data.")
        return False
    
    # Validate the value column (assumed to always be the second column in df)
    if not validate_value_column(df.iloc[:, 1]):
        st.error("The selected value column does not contain valid numerical data.")
        return False

    return True  # Both columns are valid


def convert_to_nanoseconds(value, unit, signal_freq_MHz):
    
    """Converts the given TIME / Frequency data value from the specified unit to nanoseconds."""
    
    # Check if this is a frequency to time conversion scenario
    # if signal_freq_MHz != 0:
    #     # First convert the input frequency difference to a period in seconds (T = 1 / f)
    #     if unit == 'Hz':
    #         frequency_hz = value  # Directly use the value in Hz
    #     elif unit == 'mHz':
    #         frequency_hz = value / 1_000  # Convert mHz to Hz
    #     elif unit == 'µHz':
    #         frequency_hz = value / 1_000_000  # Convert µHz to Hz
    #     elif unit == 'nHz':
    #         frequency_hz = value / 1_000_000_000  # Convert nHz to Hz
    #     else:
    #         return None  # Unknown frequency unit
        
    #     # Assuming signal_freq_MHz is the base frequency and value is the difference
    #     total_frequency_hz = signal_freq_MHz * 1_000_000 + frequency_hz
    #     period_seconds = 1 / total_frequency_hz  # Calculate the period in seconds
    #     return period_seconds * 1_000_000_000  # Convert period from seconds to nanoseconds

    # # Handle direct time unit conversions when no signal frequency is specified
    # else:
    if unit == 'ns':
        st.session_state.y_title = "Phase Data [ns]"
        st.session_state.unitmulty = 1E+9
        return value * 1_000_000_000  # No conversion needed for nanoseconds
        
    elif unit == 'µs':
        st.session_state.y_title = "Phase Data [µs]"
        st.session_state.unitmulty = 1E+6
        return value * 1_000_000  # Convert microseconds to nanoseconds
    
    elif unit == 'ms':
        st.session_state.y_title = "Phase Data [ms]"
        st.session_state.unitmulty = 1E+3
        return value * 1_000  # Convert milliseconds to nanoseconds
    
    elif unit == 's':
        st.session_state.y_title = "Phase Data [s]"
        st.session_state.unitmulty = 1
        return value  # Convert seconds to nanoseconds
    
    elif unit == "Unitless":
        st.session_state.y_title = "Fractional Frequency Offset"
        st.session_state.unitmulty = 1
        return value  # Nothing to convert 

    else:
        return None  # Unknown time unit



def normalize_values(values):

    # st.write(f"Input values: {values}")
    """
    Normalize values to integers by determining an appropriate scale.
    """
    iteration_limit = 1000  # Limit the number of iterations to prevent infinite loops
    max_scale_factor = 10**8  # Cap the scale factor to avoid overflow issues

    # Handle case where all values might be zero
    if all(val == 0 for val in values):
        return [0] * len(values)
    
    min_val = min(values)
    max_val = max(values)
    scale_factor = 1

    # Handle the scaling up for values less than 1
    if max_val < 1:
        while max_val < 1 and scale_factor < 10**iteration_limit:
            max_val *= 10
            scale_factor *= 10
            if scale_factor > max_scale_factor:
                # st.warning("Scale factor capped to avoid overflow.")
                break

    # Handle the scaling down for very large values
    elif min_val > 1:
        while min_val > 1 and scale_factor > 10**-iteration_limit:
            min_val /= 10
            scale_factor /= 10
            if scale_factor < 1/max_scale_factor:
                # st.warning("Scale factor capped to avoid overflow.")
                break

    # Apply scaling to all values, capping large integers
    normalized_values = []
    for val in values:
        scaled_value = val * scale_factor
        if abs(scaled_value) > max_scale_factor:
            st.warning("Value capped to avoid overflow.")
            scaled_value = max_scale_factor if scaled_value > 0 else -max_scale_factor
        normalized_values.append(int(scaled_value))
    
    return normalized_values


# Function to read and combine clock data while ignoring headers and special symbols

def process_file(file, timestamp_col_index, value_col_index, data_type, data_scale, frequency):
        
    def read_file_with_encoding(encoding):
        try:
            file.seek(0)
            content = StringIO(file.read().decode(encoding))
            lines = content.readlines()
            # st.write(f"Read {len(lines)} lines with encoding {encoding}")
            return lines, None
        except Exception as e:
            return None, str(e)

    # First, try reading the file with ASCII encoding
    lines, error = read_file_with_encoding('ascii')
    
    # If ASCII reading fails, try UTF-8
    if error:
        st.warning(f"Failed to read file {file.name} with ASCII encoding: {error}. Trying UTF-8 encoding.")
        lines, error = read_file_with_encoding('utf-8')
    
    # If UTF-8 reading also fails, report the error
    if error:
        st.error(f"Failed to read file {file.name} with both ASCII and UTF-8 encodings: {error}.")
        return None
    
    # Clean the lines to remove unwanted characters and handle text formatting
    cleaned_lines = []
    for i, line in enumerate(lines):
        line = line.strip()  # Remove leading/trailing whitespace
        if line and not line.lstrip().startswith(('#', '<', '"', '@')):  # Skip comment lines and empty lines
            cleaned_lines.append(line)

    if not cleaned_lines:
        st.error(f"No valid data found in file {file.name}.")
        return None  # Return None if no valid data is found

    first_valid_index = 0  # We now have cleaned lines
    # st.write(f"First valid index: {first_valid_index}, Total cleaned lines: {len(cleaned_lines)}")

    # Read the file into a pandas DataFrame from the cleaned lines
    try:
        if timestamp_col_index != 'NA':
            df = pd.read_csv(StringIO('\n'.join(cleaned_lines[first_valid_index:])), sep='\s+',
                             header=None, engine='python', usecols=[timestamp_col_index, value_col_index])
        else:  # If the timestamp is NA
            df = pd.read_csv(StringIO('\n'.join(cleaned_lines[first_valid_index:])), sep='\s+',
                             header=None, engine='python', usecols=[value_col_index])

            # Generate the timestamp values starting from 1 to the length of the values column
            df['Timestamp'] = range(1, len(df) + 1)
            df = df[['Timestamp', value_col_index]]  # Reorder columns to place 'Timestamp' first

    except Exception as e:
        st.error(f"Failed to read data from the assigned columns in file {file.name}. Error: {str(e)}")
        return None       

    if df.empty:
        st.error(f"The file {file.name} does not have enough columns based on the selected indices.")
        return None  # Return None if the DataFrame is empty

    if len(df.columns) != 2:
        st.error(f"Timestamp and value columns cannot be the same. Please select different columns.")
        return None  # Return None if column count mismatch

    # Assign column names based on user input
    if timestamp_col_index != 'NA':
        column_names = ['Timestamp', 'Value'] if timestamp_col_index < value_col_index else ['Value', 'Timestamp']
    else:
        column_names = ['Timestamp', 'Value']
    
    df.columns = column_names  # Assign correct names based on user selection

    # Convert 'Timestamp' column to integers and 'Value' column to floats
    if 'Timestamp' in df.columns:
        df['Timestamp'] = df['Timestamp'].apply(lambda x: int(float(str(x).replace(',', ''))))

    df['Value'] = df['Value'].apply(lambda x: float(str(x).replace(',', '').replace(' ', '')))  # Convert to float
    
    # Validate columns
    if timestamp_col_index != 'NA' and not validate_timestamp_column(df['Timestamp']):
        st.error(f"Invalid timestamp data in selected column {timestamp_col_index} in file {file.name}. Try to choose Timestamp column as NA assuming your data is continuous.")
        return None  # Return None if validation fails

    if not validate_value_column(df['Value']):
        st.error(f"Invalid value data in file {file.name}.")
        return None  # Return None if validation fails

    # st.write("Successfully processed file")
    # st.write(df.head())  # Display the first few rows of the DataFrame for review
    return df  # Return the processed DataFrame if everything is fine

# Function to check for column consistency
def check_column_consistency(file):
    try:
        # Assuming the file is decoded as 'utf-8', which might need to be adjusted based on the file encoding
        content = StringIO(file.getvalue().decode('utf-8'))
        sample_df = pd.read_csv(content, sep='\s+', nrows=25)  # Using regex to handle multiple spaces
        num_columns = len(sample_df.columns)
        return num_columns, None
    except Exception as e:
        return None, e
    

def process_inputs(files):
    
    data_frames = {}
    if not files:
        st.warning("Please upload the files", icon="⚠️")
        return data_frames
    # Reset the file read pointer and read the data
    # files.seek(0) #UploadedFile object in Streamlit has various attributes including .name, .size, .type, 
    # Extract and store filenames, then update the session state
    valid_filenames = [file.name for file in files]
    st.session_state.input_data['files_uploaded'] = ', '.join(valid_filenames) if valid_filenames else 'NA'
    
    num_columns, error = check_column_consistency(files[0])
    if error:
        st.error(f"An error occurred in reading the file data: {files[0].name} {error}")
        return data_frames

    if num_columns is None:
        st.error("Unable to determine the number of columns in the file.")
        return data_frames
    
    if st.session_state.timestamp_col != 'NA':
        st.session_state.timestamp_col = st.session_state.timestamp_col -1
    
   
    # Process files based on the type of file combination selected
    if st.session_state.file_combo == 'Multiple files of same clock':
        combined_df = pd.DataFrame()
        all_timestamps = set()
        last_timestamp = 0
        
        for file in files:
            df = process_file(file, st.session_state.timestamp_col, st.session_state.data_col-1, 
                            st.session_state.data_type, st.session_state.order_of_data, st.session_state.freq_scale)
            if df is not None:
                if st.session_state.timestamp_col != 'NA':
                    timestamps = set(df['Timestamp'].tolist())
                    if all_timestamps & timestamps:
                        st.error("Each file has same Timestamps, it means same clock at same time records different measurements. Each file could be of different clock, please check !.\n If you still want to continue please select TIMESTAMP column to be NA ")
                        return {}
                    all_timestamps.update(timestamps)
                elif st.session_state.timestamp_col == 'NA':
                    # Assign continuous timestamps manually
                    df['Timestamp'] = range(last_timestamp + 1, last_timestamp + len(df) + 1)
                    last_timestamp += len(df)       

               
                df.iloc[:, 1] = df.iloc[:, 1].apply(convert_to_nanoseconds, args=(st.session_state.order_of_data, st.session_state.freq_scale))
                
                combined_df = pd.concat([combined_df, df], ignore_index=True)
        
        if not combined_df.empty:
            data_frames['combined'] = combined_df
            # st.write(data_frames)

    else:  # Each file is a different clock
       
        for file in files:
            df = process_file(file, st.session_state.timestamp_col, st.session_state.data_col-1, 
                            st.session_state.data_type, st.session_state.order_of_data, st.session_state.freq_scale)
            
            if df is not None:

                # st.write("Initial DataFrame after processing:")
                # st.write(df.head())

                # Check if all values in 'Value' column are None or empty strings
                if df['Value'].apply(lambda x: x is None or x == '').all():
                    st.warning(f"The data in the file {file.name} has None values")
                # st.write("Danger")

                # Convert values to nanoseconds if there are valid entries
                if not df['Value'].isnull().all():
                
                    df.iloc[:, 1] = df.iloc[:, 1].apply(convert_to_nanoseconds, args=(st.session_state.order_of_data, st.session_state.freq_scale))
                    # st.write("DataFrame after converting values to nanoseconds:")
                    # st.write(df.head())

                else:
                    st.warning(f"Skipping conversion to nanoseconds for file {file.name} due to all NaN values.")


                file_key = file.name.split('.')[0]  # Using file name without extension as the key

                data_frames[file_key] = df
    
                # st.write(f"DataFrames now contains keys: {list(data_frames.keys())}")
    # st.write(data_frames)
    return data_frames
    


# Define a function to create an empty DataFrame if not already in session state
def initialize_display_df():
    if 'df_display' not in st.session_state or st.session_state.df_display.empty:
        st.session_state.df_display = pd.DataFrame({
            "Files uploaded": pd.Series([], dtype=str),
            "Sample_data": pd.Series([], dtype=str),
            "Choose Clock": pd.Series([], dtype=bool),
            "Clock Name": pd.Series([], dtype=str),
        })

# Define a default state or reset state function
def initialize_state():
    """Initialize or reset the session state."""
    st.session_state.input_data = {
        'files_uploaded': 'NA',
        'data_type': 'NA',
        'units_data': 'NA',
        'signal_frequency': 'NA',
        'file_combo': 'NA',
        'timestamp_col': 'NA',
        'data_col': 'NA',
        'tau0': 'NA'
    }
    st.session_state.clks_data = False
    st.session_state.proceed = False  # Reset the proceed flag
    st.session_state.data_type = None  # Initialize data_type
    st.session_state.valid_filenames = [] # List of valid file names 
    st.session_state.files_uploaded = None
    st.session_state.data_loaded = False
    initialize_display_df()  # Initialize the empty DataFrame when the app starts
    initialize_out_display()
    st.session_state.clk_sel = False # Clock selected from the raw data 
    st.session_state.total_data = {}
    st.session_state.data = {}
    st.session_state.clk_to_analyse= pd.DataFrame()
    st.session_state.clk_filename = None
    st.session_state.selected_points = []
    st.session_state.trend_selection = {}
    st.session_state.checkbox_states = {}
    st.session_state['filtered_data'] = pd.DataFrame(columns=["Timestamp", "Value"])
    st.session_state['clock_ranges'] = {}
    st.session_state['clk_data_full'] = pd.DataFrame(columns=["Timestamp", "Value"])
    st.session_state['offset_selection'] = {}
    st.session_state['total_data'] = {}
    st.session_state['clk_sel'] = False
    st.session_state.trend_slopes = {}
    st.session_state.trend_intercepts = {}
    st.session_state.trend_coeffs = {}
    st.session_state['processed_data'] = pd.DataFrame()
    st.session_state.detrended_data = {}
    st.session_state.offset_removed_data = st.session_state.processed_data.copy()
    st.session_state.initial_offset_removed_data = st.session_state.processed_data.copy()
    st.session_state.outlier_selection= {}
    st.session_state.stability_results = {
        "ADEV": {},
        "MDEV": {},
        "OADEV": {},
        "TDEV": {}
    }

# Define a function to create an empty DataFrame if not already in session state
def initialize_out_display():
    if 'out_display' not in st.session_state or st.session_state.out_display.empty:
        st.session_state.out_display = pd.DataFrame({
            "Files uploaded": pd.Series([], dtype=str),
            "Data Type": pd.Series([], dtype=str),
            "Clock Name": pd.Series([], dtype=str),
            "Data Range": pd.Series([], dtype=str),
            "Measurement Interval [s]": pd.Series([], dtype=str),
            "Detrend": pd.Series([], dtype=str),
            "Offset Removed": pd.Series([], dtype=str),
            "Outlier Filtered": pd.Series([], dtype=str),
            "Smoothed": pd.Series([], dtype=str),
            "ADEV": pd.Series([], dtype=object),
            "MDEV": pd.Series([], dtype=object),
            "OADEV": pd.Series([], dtype=object),
            "TDEV": pd.Series([], dtype=object),
        })


# Function to initialize data for each clock and tab
def initialize_clock_data(clock_name):
    tabs = ["raw_data", "data_range", "detrend", "offset", "outlier", "smoothing", "stability"]
    if clock_name not in st.session_state.data:
        st.session_state.data[clock_name] = {tab: pd.DataFrame() for tab in tabs}

def get_latest_data(clock_name, current_tab):
    tabs_order = ['raw_data', 'data_range', 'detrend', 'offset', 'outlier', 'smoothing', 'stability']
    current_index = tabs_order.index(current_tab)
    for tab in tabs_order[:current_index + 1][::-1]:
        if not st.session_state.data[clock_name][tab].empty:
            return st.session_state.data[clock_name][tab]
    return pd.DataFrame(columns=["Timestamp", "Value"])  # Default empty DataFrame


def update_action(clock_name, action, value):
    if clock_name == "Combine Clocks":
        return  # Skip "Combine Clocks"

    # Check if the clock already exists in the DataFrame
    idx = st.session_state.out_display[st.session_state.out_display['Clock Name'] == clock_name].index

    if not idx.empty:
        idx = idx[0]
        # Update the specific action for the existing clock entry
        st.session_state.out_display.at[idx, action] = value
    else:
        # Find the file name corresponding to the clock name using the mapping
        matching_rows = st.session_state.df_display[st.session_state.df_display['Clock Name'] == clock_name]
        if not matching_rows.empty:
            file_uploaded = matching_rows['Files uploaded'].values[0]
        else:
            file_uploaded = 'NA'  # Handle the case where the clock name is not found

        # Initialize new row with 'NA' for all actions except the one being updated
        new_row = {
            "Files uploaded": file_uploaded,
            "Data Type": st.session_state.input_data.get('data_type', 'NA'),
            "Clock Name": clock_name,
            "Measurement Interval [s]": st.session_state.tau0 , 
            "Data Range": 'NA',
            "Detrend": 'NA',
            "Offset Removed": 'NA',
            "Outlier Filtered": 'NA',
            "Smoothed": 'NA',
            "ADEV": 'NA',
            "MDEV": 'NA',
            "OADEV": 'NA',
            "TDEV": 'NA',
        }
        new_row[action] = value

        # Append the new row to the DataFrame
        st.session_state.out_display = pd.concat([st.session_state.out_display, pd.DataFrame([new_row])], ignore_index=True)


def update_stability_results(clock_name, analysis_type, tau_values, dev_values):
    
    if 'stability_results' not in st.session_state:
        st.session_state.stability_results = {}
    if analysis_type not in st.session_state.stability_results:
        st.session_state.stability_results[analysis_type] = {}
    
    formatted_results = [f"{int(tau)},{dev:.2e}" for tau, dev in zip(tau_values, dev_values) if not np.isnan(tau) and not np.isnan(dev)]
    st.session_state.stability_results[analysis_type][clock_name] = formatted_results

def render_html_table(df, styles):
    # Convert the DataFrame to HTML without the index
    html = df.to_html(escape=False, index=False, classes='dataframe', border=0, header=False)

    # Combine the CSS styles and HTML table
    full_html = f"<style>{styles}</style>{html}"
    return full_html


def convert_df_to_text(df, delimiter='; '):
    # Replace <br> tags with the specified delimiter
    df = df.applymap(lambda x: x.replace('<br>', delimiter) if isinstance(x, str) else x)
    
    output = StringIO()
    df.to_csv(output, sep=' ', index=False, header=False)
    return output.getvalue()


# Define the CSS styles for the table
styles = """
.container {
    width: 100%;
    overflow-x: auto;
    max-height: 500px;
    display: flex;
    justify-content: center;  /* Center the table horizontally */
}
.dataframe {
    width: 100%;
    border-collapse: collapse;
    max-height: 500px;
    # display: block;
}
.dataframe th, .dataframe td {
    border: 1px solid #ddd;
    padding: 8px;
    text-align: left;
}
.dataframe th {
    background-color: #f2f2f2;
    font-weight: bold;
}
.dataframe tr:nth-child(even) {
    background-color: #f2f2f2;
}
.dataframe tr:nth-child(odd) {
    background-color: #ffffff;
}
.dataframe thead tr {
    position: sticky;
    top: 0;
    background-color: #4CAF50;
    color: white;
    z-index: 1;
}
.dataframe tbody tr:first-child {
    background-color: #4CAF50;
    color: white;
}
"""


def format_scientific(x):
    try:
        return f"{float(x):.2e}"
    except (ValueError, TypeError):
        return x

def format_scientific2(value):
    # return f"{value:.2e}" if isinstance(value, (float, int)) else value
    try:
        return f"{value * st.session_state.unitmulty:.2e}" if isinstance(value, (float, int)) else value
    except Exception as e:
        st.write(f"Error formatting value {value}: {e}")
        return value
    

def transform_and_display(selected_clock_names):
    if 'out_display' not in st.session_state:
        return

    # Filter out clocks that are not in the selected clock names
    out_display_filtered = st.session_state.out_display[st.session_state.out_display['Clock Name'].isin(selected_clock_names)]

    # Extract basic info columns and prepare the base DataFrame
    basic_info = out_display_filtered.drop(columns=["ADEV", "MDEV", "OADEV", "TDEV"]).set_index('Clock Name').T
    basic_info.insert(0, '', basic_info.index)

    # Initialize a list to collect final DataFrame rows
    final_rows = [basic_info.columns.tolist()] + basic_info.values.tolist()

    # Process each stability type and append results
    for analysis_type in ["ADEV", "MDEV", "OADEV", "TDEV"]:
        # Append the analysis type row
        analysis_done_row = [f"{analysis_type} Analysis"] + [""] * (basic_info.shape[1] - 1)
        final_rows.append(analysis_done_row)

        # Collect stability results
        stability_results = []
        max_length = 0
        for clock in basic_info.columns[1:]:
            if clock in st.session_state.stability_results.get(analysis_type, {}):
                result_list = st.session_state.stability_results[analysis_type][clock]
                stability_results.append(result_list)
                max_length = max(max_length, len(result_list))
            else:
                stability_results.append([])

        # Append Tau values and results in rows
        for i in range(max_length):
            row_data = []
            for result_list in stability_results:
                if i < len(result_list):
                    row_data.append(result_list[i])  # Combined tau and stability value
                else:
                    row_data.append("")
            final_rows.append([""] + row_data)  # Add an empty first cell to match the final DataFrame structure

    # Ensure all rows have the same number of columns
    max_columns = max(len(row) for row in final_rows)
    final_rows = [row + [""] * (max_columns - len(row)) for row in final_rows]

    # Create a DataFrame from the final rows
    final_df = pd.DataFrame(final_rows)

    html_table = render_html_table(final_df, styles)
    # st.markdown(html_table, unsafe_allow_html=True)
    st.markdown(f'<div class="container">{html_table}</div>', unsafe_allow_html=True)
    st.markdown("")
     # Convert DataFrame to space-separated text
    text_data = convert_df_to_text(final_df)
    st.download_button(
        label="Download data as text file",
        data=text_data,
        file_name='complete_data.txt',
        mime='text/plain'
    )


# Function to parse timestamps
def parse_timestamp(timestamp):
    try:
        return float(timestamp)
    except ValueError:
        pass
    datetime_formats = ['%Y-%m-%d %H:%M:%S', '%H:%M:%S']
    for fmt in datetime_formats:
        try:
            dt = datetime.strptime(timestamp, fmt)
            if fmt == '%H:%M:%S':
                today = datetime.today().date()
                return datetime.combine(today, dt.time())
            return dt
        except ValueError:
            continue
    raise ValueError(f"Unsupported timestamp format: {timestamp}")

# Function to calculate x-intervals
def x_interval(data):
    x = [parse_timestamp(ts) for ts in data['Timestamp']]
    dx = []
    for i in range(1, len(x)):
        try:
            if isinstance(x[i], float) and isinstance(x[i-1], float):
                diff = x[i] - x[i-1]
            elif isinstance(x[i], datetime) and isinstance(x[i-1], datetime):
                diff = (x[i] - x[i-1]).total_seconds()
            else:
                raise TypeError("Mismatch in timestamp types or unsupported type")
            
            if diff != 0:
                dx.append(diff)
        except TypeError as e:
            print(f"TypeError: {e}")
            continue  # Skip this interval if there is a type error

    # Ensure that x and dx have the same length
    x = x[1:len(dx) + 1]
    st.write(x)

    return x, dx

# Function to filter data
def update_filtered_data(start, end):
    filtered_data = st.session_state['clk_data_full'][
        (st.session_state['clk_data_full']["Timestamp"] >= start) &
        (st.session_state['clk_data_full']["Timestamp"] <= end)
    ]
    st.session_state['filtered_data'] = filtered_data


# def generate_checkbox_state(row_index):
#     # Check if checkbox_states exist in session state
#     if 'checkbox_states' in st.session_state:
#         # Check if the list index is within bounds
#         if row_index < len(st.session_state.checkbox_states):
#             return st.session_state.checkbox_states[row_index]
#     return False  # Default to unchecked


def generate_checkbox_state(clock_name):
    # Check if checkbox_states exist in session state
    if 'checkbox_states' in st.session_state:
        # Return the checkbox state for the given clock name
        return st.session_state.checkbox_states.get(clock_name, False)
    return False  # Default to unchecked

# Initialize the state for clock ranges if not already initialized
def initialize_clock_ranges():
    if 'clock_ranges' not in st.session_state:
        st.session_state.clock_ranges = {}
    for index, row in st.session_state.df_display.iterrows():
        clock_name = row["Clock Name"]
        if clock_name not in st.session_state.clock_ranges:
            clk_analysis = get_latest_data(clock_name, 'data_range')
            st.session_state.clock_ranges[clock_name] = {
                'start_range': clk_analysis["Timestamp"].iloc[0],
                'end_range': clk_analysis["Timestamp"].max()
            }

def update_range(clock_name, range_type, new_value):
    """
    This function updates the horizontal axis range for a specific clock and triggers data filtering and plot updates.

    Args:
        clock_name (str): Name of the clock to update.
        range_type (str): "start" or "end" indicating which range to update.
        new_value (str): The new value entered by the user.
    """
    if new_value:
        try:
            new_value = float(new_value)
            # Update session state based on range type
            if range_type == "start":
                st.session_state.clock_ranges[clock_name]["start_range"] = new_value
            elif range_type == "end":
                st.session_state.clock_ranges[clock_name]["end_range"] = new_value
            else:
                st.error(f"Invalid range type: {range_type}" )
                return  # Exit function if invalid range type

            # Ensure end range is greater than start range
            if st.session_state.clock_ranges[clock_name]["start_range"] < st.session_state.clock_ranges[clock_name]["end_range"]:
                # Get filtered data based on updated range
                clk_analysis = get_latest_data(clock_name, 'data_range')
                st.session_state['filtered_data'] = clk_analysis[(clk_analysis["Timestamp"] >= st.session_state.clock_ranges[clock_name]["start_range"]) & (clk_analysis["Timestamp"] <= st.session_state.clock_ranges[clock_name]["end_range"])].copy()

                # Update session state with new values
                st.session_state[f"start_float_{clock_name}"] = str(st.session_state.clock_ranges[clock_name]["start_range"])
                st.session_state[f"end_float_{clock_name}"] = str(st.session_state.clock_ranges[clock_name]["end_range"])

                # Update plot with filtered data
                # fig = create_plots(st.session_state['filtered_data']['Timestamp'], st.session_state['filtered_data']['Value'])
                # st.plotly_chart(fig, use_container_width=True)

                # Update action log (if applicable)
                update_action(clock_name, 'Data Range', f"Start: {st.session_state.clock_ranges[clock_name]['start_range']}, End: {st.session_state.clock_ranges[clock_name]['end_range']}")
            else:
                st.error("End Range must be greater than Start Range")
        except ValueError as e:
            st.error(f"Invalid input: {e}")


# Function to create and return an empty DataFrame with the same structure
def create_empty_df():
    return pd.DataFrame({
        "Files uploaded": pd.Series([], dtype=str),
        "Clock Name": pd.Series([], dtype=str),
        "Sample_data": pd.Series([], dtype=str),
        "Choose Clock": pd.Series([], dtype=bool)
    })

# Initialize or adjust session state for checkbox states
if 'df_display' not in st.session_state:
    st.session_state.df_display = create_empty_df()


def initialize_detrend_info():
    if 'trend_selection' not in st.session_state:
        st.session_state.trend_selection = {}
    if 'trend_slopes' not in st.session_state:
        st.session_state.trend_slopes = {}
    if 'trend_coeffs' not in st.session_state:
        st.session_state.trend_coeffs = {}

    for index, row in st.session_state.df_display.iterrows():
        clock_name = row["Clock Name"]
        if clock_name not in st.session_state.trend_selection:
            st.session_state.trend_selection[clock_name] = "None"
        if clock_name not in st.session_state.trend_slopes:
            st.session_state.trend_slopes[clock_name] = None
        if clock_name not in st.session_state.trend_coeffs:
            st.session_state.trend_coeffs[clock_name] = None
        if clock_name not in st.session_state.trend_intercepts:
            st.session_state.trend_intercepts[clock_name] = None

initialize_detrend_info()

def initialize_offset_info():
    if 'offset_selection' not in st.session_state:
        st.session_state.offset_selection = {}
    if 'offset_means_before' not in st.session_state:
        st.session_state.offset_means_before = {}
    if 'offset_means_after' not in st.session_state:
        st.session_state.offset_means_after = {}

    for index, row in st.session_state.df_display.iterrows():
        clock_name = row["Clock Name"]
        if clock_name not in st.session_state.offset_selection:
            st.session_state.offset_selection[clock_name] = "None"
        if clock_name not in st.session_state.offset_means_before:
            st.session_state.offset_means_before[clock_name] = None
        if clock_name not in st.session_state.offset_means_after:
            st.session_state.offset_means_after[clock_name] = None

initialize_offset_info()


def initialize_outlier_info():
    if 'outlier_selection' not in st.session_state:
        st.session_state.outlier_selection = {}
    if 'std_threshold' not in st.session_state:
        st.session_state.std_threshold = {}
    if 'outlier_data' not in st.session_state:
        st.session_state.outlier_data = {}

    for index, row in st.session_state.df_display.iterrows():
        clock_name = row["Clock Name"]
        if clock_name not in st.session_state.outlier_selection:
            st.session_state.outlier_selection[clock_name] = "None"
        if clock_name not in st.session_state.std_threshold:
            st.session_state.std_threshold[clock_name] = 50.0
        if clock_name not in st.session_state.outlier_data:
            st.session_state.outlier_data[clock_name] = pd.DataFrame()

initialize_outlier_info()


def initialize_smoothing_info():
    if 'smoothing_method' not in st.session_state:
        st.session_state.smoothing_method = {}
    if 'window_size' not in st.session_state:
        st.session_state.window_size = {}
    if 'smoothed_data' not in st.session_state:
        st.session_state.smoothed_data = {}

    for index, row in st.session_state.df_display.iterrows():
        clock_name = row["Clock Name"]
        if clock_name not in st.session_state.smoothing_method:
            st.session_state.smoothing_method[clock_name] = 'None'
        if clock_name not in st.session_state.window_size:
            st.session_state.window_size[clock_name] = 'N/A'
        if clock_name not in st.session_state.smoothed_data:
            st.session_state.smoothed_data[clock_name] = pd.DataFrame()

initialize_smoothing_info()

def initialize_stability_info():
    if 'stability_results' not in st.session_state:
        st.session_state.stability_results = {}

    for index, row in st.session_state.df_display.iterrows():
        clock_name = row["Clock Name"]
        if clock_name not in st.session_state.stability_results:
            st.session_state.stability_results[clock_name] = {}

initialize_stability_info()



def create_plots(timestamps, data):
# """
# This function creates a plotly figure with various subplots for data visualization.

# Args:
#     timestamps: A list or array containing the timestamps for the data.
#     data: A list or array containing the data values.

# Returns:
#     A plotly figure object.
# """

        # Create a DataFrame to handle NaN values
    df = pd.DataFrame({'Timestamp': timestamps, 'Value': data})

    # Filter out rows where 'Value' is NaN
    df = df.dropna(subset=['Value'])

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=("", "Histogram", "Sampling Interval", ""),
        specs=[[{'type': 'scatter'}, {'type': 'histogram'}],
               [{'type': 'scatter'}, None]],
        column_widths=[0.85, 0.15],
        row_heights=[0.80, 0.20]
    )

    trace = go.Scatter(x=df['Timestamp'], y=df['Value'], mode='markers', name='Original')
    fig.add_trace(trace, row=1, col=1)
    fig.add_trace(go.Histogram(y=df['Value'], nbinsx=50, name='Histogram'), row=1, col=2)

    # Calculate sampling intervals
    intervals = df['Timestamp'].diff().fillna(0)[1:]  # Exclude the first interval
    interval_timestamps = df['Timestamp'][1:]  # Exclude the first timestamp to match intervals

    fig.add_trace(
        go.Scatter(
            x=interval_timestamps,
            y=intervals,
            mode='lines+markers',
            name='Sampling Interval',
            marker=dict(color='royalblue'),
            line=dict(dash='dot')
        ),
        row=2, col=1
    )

    fig.update_xaxes(tickformat=".1f", tickfont=dict(size=14, color="black"), exponentformat='none', row=2, col=1)
    fig.update_xaxes(tickformat=".1f", tickfont=dict(size=14, color="black"), exponentformat='none', row=1, col=1)

    # Determine y-axis title based on tau0 value
    tau0 = st.session_state.input_data['tau0']
    y_axis_title = "Days" if tau0 >= 86400 else "Seconds"
    fig.update_yaxes(title_text=y_axis_title, row=2, col=1)  # Set y-axis label for the subplot in row 2, col 1

    fig.update_layout(
        title="Clock Data",
        xaxis_title="MJD",
        yaxis_title=st.session_state.y_title,
        yaxis=dict(tickmode='auto', nticks=10),
        showlegend=False,
        xaxis=dict(tickformat=".1f", tickfont=dict(size=14, color="black"), exponentformat='none'),
        height=600
    )

    return fig

# Function to remove trend
def remove_trend(data, trend_type):
    x = np.arange(len(data))
    if trend_type == 'Linear':
        p = np.polyfit(x, data, 1)
        trend = np.polyval(p, x)
        return data - trend, p[0], [p[1]]  # Return residuals and slope and intercept
    elif trend_type == 'Quadratic':
        p = np.polyfit(x, data, 2)
        trend = np.polyval(p, x)
        return data - trend, None, p  # Return residuals and coefficients
    else:
        trend = np.zeros_like(data)
        return data - trend, None, None  # Return residuals
        

def remove_offset(data):
    mean_value = np.mean(data)
    offset_removed_data = data - mean_value
    return offset_removed_data, mean_value


def remove_outliers(data, std_threshold, value_col="Value"):
    mean = np.mean(data[value_col])
    std_dev = np.std(data[value_col])
    filtered_data = data[np.abs(data[value_col] - mean) <= std_threshold * std_dev]
    return filtered_data, std_dev, np.std(filtered_data[value_col])


# Symmetric Moving average filter for both even and odd window sizes
def moving_average(data, window_size):
    half_window = window_size // 2
    if window_size % 2 == 0:
        # Even window size
        extended_data = np.pad(data, (half_window - 1, half_window), mode='reflect')
    else:
        # Odd window size
        extended_data = np.pad(data, (half_window, half_window), mode='reflect')
    
    smoothed_data = np.convolve(extended_data, np.ones(window_size) / window_size, mode='valid')
    return smoothed_data

# Calculate moving average
def smoothing_overlap(data, window_size, timestamps):
    smoothed_data = moving_average(data, window_size)
    half_window = window_size // 2
    if window_size % 2 == 0:
        valid_timestamps = timestamps[half_window - 1: -half_window]
    else:
        valid_timestamps = timestamps[half_window: -half_window]

    # Ensure lengths are the same
    valid_timestamps = valid_timestamps[:len(smoothed_data)]
    smoothed_data = smoothed_data[:len(valid_timestamps)]
    
    return valid_timestamps, smoothed_data

# Calculate moving average for overlapping case
def smoothing_nonoverlap(data, window_size, timestamps):
    n = len(data)
    num_segments = n // window_size

    # Only consider full segments
    data = data[:num_segments * window_size]
    timestamps = timestamps[:num_segments * window_size]

    # Reshape data to have num_segments rows, each of length window_size
    reshaped_data = data.reshape(num_segments, window_size)
    reshaped_timestamps = timestamps.reshape(num_segments, window_size)

    # Compute the mean for each segment
    smoothed_data = reshaped_data.mean(axis=1)
    valid_timestamps = reshaped_timestamps.mean(axis=1)

    return valid_timestamps, smoothed_data

def read_data(data):
    return data

def plot_mdev(data,sampling_int,data_type, label=None, color='orange', marker='o'):
    # max_tau_exponent = int(np.floor(np.log2(len(data)))) - 2
    # tau_values = [2**i for i in range(max_tau_exponent + 1)]
    max_tau_exponent = int(np.floor(np.log2(len(data) * sampling_int))) - 2
    tau_values = [2**i * sampling_int for i in range(max_tau_exponent + 1) if 2**i * sampling_int < (len(data)*sampling_int)]
    
    # data_type = "phase"
    # taus, ad, ade, ns = adev(data, (1/sampling_int), taus=tau_values)


    tau_used, md, mderr, ns = mdev(data,(1/sampling_int), data_type, tau_values)

    plt.loglog(tau_used, md, marker=marker, color=color, label=label)
    plt.xlabel('Tau (s)')
    plt.ylabel('Modified Allan Deviation')
    plt.title('Modified Allan Deviation (MDEV)')
    plt.legend()
    plt.grid(True, which='both', linestyle='-', linewidth=0.8, color='gray', alpha=0.5)
    plt.minorticks_on()
    plt.grid(which='minor', linestyle=':', linewidth='0.5', color='gray')
    return tau_used, md

def plot_tdev(data, sampling_int, data_type, label=None, color='blue', marker='*'):
    # max_tau_exponent = int(np.floor(np.log2(len(data)))) - 2
    # tau_values = [2**i for i in range(max_tau_exponent + 1)]

    max_tau_exponent = int(np.floor(np.log2(len(data) * sampling_int))) - 2
    tau_values = [2**i * sampling_int for i in range(max_tau_exponent + 1) if 2**i * sampling_int < (len(data)*sampling_int)]
                
    # data_type = "phase"
    taus, td, tde, ns = tdev(data, (1/sampling_int), data_type,  tau_values)

    plt.loglog(taus, td, marker=marker, color=color, label=label)
    plt.xlabel('Tau (s)')
    plt.ylabel('Time Deviation [s]')
    plt.title('Time Deviation (TDEV)')
    plt.legend()
    plt.grid(True, which='both', linestyle='-', linewidth=0.8, color='gray', alpha=0.5)
    plt.minorticks_on()
    plt.grid(which='minor', linestyle=':', linewidth='0.5', color='gray')
    return taus, td

def plot_oadev(data, sampling_int, data_type, label=None, color='red', marker='^'):
    # max_tau_exponent = int(np.floor(np.log2(len(data)))) - 2
    # tau_values = [2**i for i in range(max_tau_exponent + 1)]

    max_tau_exponent = int(np.floor(np.log2(len(data) * sampling_int))) - 2
    tau_values = [2**i * sampling_int for i in range(max_tau_exponent + 1) if 2**i * sampling_int < (len(data)*sampling_int)]
                
    # data_type = "phase"

    taus, ad, ade, ns = oadev(data, (1/sampling_int), data_type, taus=tau_values)
    plt.loglog(taus, ad, marker=marker, color=color, label=label)
    plt.xlabel('Tau (s)')
    plt.ylabel('Overlapping Allan Deviation')
    plt.title('Overlapping Allan Deviation (OADEV)')
    plt.legend()
    plt.grid(True, which='both', linestyle='-', linewidth=0.8, color='gray', alpha=0.5)
    plt.minorticks_on()
    plt.grid(which='minor', linestyle=':', linewidth='0.5', color='gray')
    return taus, ad

def plot_adev(data, sampling_int, data_type, label=None, color='green', marker='s'):
    # max_tau_exponent = int(np.floor(np.log2(len(data)))) - 2
    # tau_values = [2**i for i in range(max_tau_exponent + 1)]
    # Calculate the maximum tau exponent so that tau_values remain less than the length of data
    max_tau_exponent = int(np.floor(np.log2(len(data) * sampling_int))) - 2
    tau_values = [2**i * sampling_int for i in range(max_tau_exponent + 1) if 2**i * sampling_int < (len(data)*sampling_int)]
    

    taus, ad, ade, ns = adev(data, (1/sampling_int), data_type,taus=tau_values)
    plt.loglog(taus, ad, marker=marker, color=color, label=label)
    plt.xlabel('Tau (s)')
    plt.ylabel('Allan Deviation')
    plt.title('Allan Deviation (ADEV)')
    plt.legend()
    plt.grid(True, which='both', linestyle='-', linewidth=0.8, color='gray', alpha=0.5)
    plt.minorticks_on()
    plt.grid(which='minor', linestyle=':', linewidth='0.5', color='gray')
    return taus, ad


def mdev(data, rate=1.0, data_type="phase", taus=None):
    """  Modified Allan deviation.
    Used to distinguish between White and Flicker Phase Modulation.

    .. math::

        \\sigma^2_{MDEV}(m\\tau_0) = { 1 \\over 2 (m \\tau_0 )^2 (N-3m+1) }
        \\sum_{j=1}^{N-3m+1} \\left[
        \\sum_{i=j}^{j+m-1} {x}_{i+2m} - 2x_{i+m} + x_{i} \\right]^2

    Parameters
    ----------
    data: np.array
        Input data. Provide either phase or frequency (fractional,
        adimensional).
    rate: float
        The sampling rate for data, in Hz. Defaults to 1.0
    data_type: {'phase', 'freq'}
        Data type, i.e. phase or frequency. Defaults to "phase".
    taus: np.array
        Array of tau values, in seconds, for which to compute statistic.
        Optionally set taus=["all"|"octave"|"decade"] for automatic
        tau-list generation.

    Returns
    -------
    (taus2, md, mde, ns): tuple
        Tuple of values
    taus2: np.array
        Tau values for which td computed
    md: np.array
        Computed mdev for each tau value
    mde: np.array
        mdev errors
    ns: np.array
        Values of N used in each mdev calculation

    References
    ----------
    * NIST [SP1065]_ eqn (14), page 17.
    * http://www.leapsecond.com/tools/adev_lib.c
    """
    phase = input_to_phase(data, rate, data_type)
    
    (phase, ms, taus_used) = tau_generator(phase, rate, "mdev", taus=taus)
    data, taus = np.array(phase), np.array(taus)

    md = np.zeros_like(ms, dtype=np.float64)
    mderr = np.zeros_like(ms, dtype=np.float64)
    ns = np.zeros_like(ms)
    
    # this is a 'loop-unrolled' algorithm following
    # http://www.leapsecond.com/tools/adev_lib.c
    for idx, m in enumerate(ms):
        m = int(m)  # without this we get: VisibleDeprecationWarning:
        # using a non-integer number instead of an integer
        # will result in an error in the future
        tau = taus_used[idx]

        # First loop sum
        d0 = phase[0:m]
        d1 = phase[m:2*m]
        d2 = phase[2*m:3*m]
        e = min(len(d0), len(d1), len(d2))
        v = np.sum(d2[:e] - 2*d1[:e] + d0[:e])
        s = v * v

        # Second part of sum
        d3 = phase[3*m:]
        d2 = phase[2*m:]
        d1 = phase[1*m:]
        d0 = phase[0:]

        e = min(len(d0), len(d1), len(d2), len(d3))
        n = e + 1

        v_arr = v + np.cumsum(d3[:e] - 3 * d2[:e] + 3 * d1[:e] - d0[:e])

        s = s + np.sum(v_arr * v_arr)
        s /= 2.0 * m * m * tau * tau * n
        s = np.sqrt(s)


        md[idx] = s
        mderr[idx] = (s / np.sqrt(n))
        ns[idx] = n
        

    
    return remove_small_ns(taus_used, md, mderr, ns)

def tdev(data, rate=1.0, data_type="phase", taus=None):
    (taus_used, md, mde, ns) = mdev(data, rate, data_type,  taus)

    td = taus_used * md / np.sqrt(3.0)
    tde = td / np.sqrt(ns)

    return taus_used, td, tde, ns

def trim_data(x):
    """Trim leading and trailing NaNs from dataset

    This is done by browsing the array from each end and store the index of the
    first non-NaN in each case, the return the appropriate slice of the array
    """
    # Find indices for first and last valid data
    first = 0
    while np.isnan(x[first]):
        first += 1
    last = len(x)
    while np.isnan(x[last - 1]):
        last -= 1
    return x[first:last]

def frequency2phase(freqdata, rate):
    """ integrate fractional frequency data and output phase data

    Parameters
    ----------
    freqdata: np.array
        Data array of fractional frequency measurements (nondimensional)
    rate: float
        The sampling rate for phase or frequency, in Hz

    Returns
    -------
    phasedata: np.array
        Time integral of fractional frequency data, i.e. phase (time) data
        in units of seconds.
        For phase in units of radians, see phase2radians()
    """
    dt = 1.0 / float(rate)
    # Protect against NaN values in input array (issue #60)
    # Reintroduces data trimming as in commit 503cb82
    freqdata = trim_data(freqdata)
    # Erik Benkler (PTB): Subtract mean value before cumsum in order to
    # avoid precision issues when we have small frequency fluctuations on
    # a large average frequency
    freqdata = freqdata - np.nanmean(freqdata)
    phasedata = np.cumsum(freqdata) * dt
    phasedata = np.insert(phasedata, 0, 0)  # FIXME: why do we do this?
    # so that phase starts at zero and len(phase)=len(freq)+1 ??
    return phasedata

def input_to_phase(data, rate, data_type):
   
    """ Take either phase or frequency as input and return phase
    """
    if data_type == "phase":
        return data
    elif data_type == "frequency":
        return frequency2phase(data, rate)
    else:
        raise Exception("unknown data_type: " + data_type)
    

def calc_adev_phase(phase, rate, mj, stride):
    """  Main algorithm for adev() (stride=mj) and oadev() (stride=1)

    Parameters
    ----------
    phase: np.array
        Phase data in seconds.
    rate: float
        The sampling rate for phase or frequency, in Hz
    mj: int
        averaging factor, we evaluate at tau = m*tau0
    stride: int
        Size of stride

    Returns
    -------
    (dev, deverr, n): tuple
        Array of computed values.

    Notes
    -----
    stride = mj for nonoverlapping Allan deviation
    stride = 1 for overlapping Allan deviation

    * NIST [SP1065]_ eqn (7) and (11) page 16
    """
    mj = int(mj)
    stride = int(stride)
    d2 = phase[2 * mj::stride]
    d1 = phase[1 * mj::stride]
    d0 = phase[::stride]

    n = min(len(d0), len(d1), len(d2))

    if n == 0:
        RuntimeWarning("Data array length is too small: %i" % len(phase))
        n = 1

    v_arr = d2[:n] - 2 * d1[:n] + d0[:n]
    s = np.sum(v_arr * v_arr)

    dev = np.sqrt(s / (2.0*n)) / mj*rate
    deverr = dev / np.sqrt(n)

    return dev, deverr, n


def tau_generator(data, rate, dev, taus=None, v=False, even=False, maximum_m=-1):
    

    """ pre-processing of the tau-list given by the user (Helper function)

    Does sanity checks, sorts data, removes duplicates and invalid values.
    Generates a tau-list based on keywords 'all', 'decade', 'octave'.
    Uses 'octave' by default if no taus= argument is given.

    Parameters
    ----------
    data: np.array
        data array
    rate: float
        Sample rate of data in Hz. Time interval between measurements
        is 1/rate seconds.
    taus: np.array
        Array of tau values for which to compute measurement.
        Alternatively one of the keywords: "all", "octave", "decade".
        Defaults to "octave" if omitted.

        +----------+--------------------------------+
        | keyword  |   averaging-factors            |
        +==========+================================+
        | "all"    |  1, 2, 3, 4, ..., len(data)    |
        +----------+--------------------------------+
        | "octave" |  1, 2, 4, 8, 16, 32, ...       |
        +----------+--------------------------------+
        | "decade" |  1, 2, 4, 10, 20, 40, 100, ... |
        +----------+--------------------------------+
        | "log10"  |  approx. 10 points per decade  |
        +----------+--------------------------------+
    v: bool
        verbose output if True
    even: bool
        require even m, where tau=m*tau0, for Theo1 statistic
    maximum_m: int
        limit m, where tau=m*tau0, to this value.
        used by mtotdev() and htotdev() to limit maximum tau.

    Returns
    -------
    (data, m, taus): tuple
        List of computed values
    data: np.array
        Data
    m: np.array
        Tau in units of data points
    taus: np.array
        Cleaned up list of tau values
    """
    # st.write(f"Rate: {rate}")
    # st.write(f"Data input: {data}")
    # st.write(f"taus: {taus}")

    if rate == 0:
        raise RuntimeError("Warning! rate==0")

    if taus is None:  # empty or no tau-list supplied
        taus = "octave"  # default to octave
        # st.write("tau input is None")
    elif isinstance(taus, list) and taus == []:  # empty list
        taus = "octave"

    # numpy array or non-empty list detected first
    if isinstance(taus, np.ndarray) or isinstance(taus, list) and len(taus):
        pass
    elif taus == "all":  # was 'is'
        taus = (1.0/rate)*np.linspace(1.0, len(data), len(data))
    elif taus == "octave":
        maxn = np.floor(np.log2(len(data)))
        taus = (1.0/rate)*np.logspace(0, int(maxn), int(maxn+1), base=2.0)
        # st.write(f"Taus CREATED: {taus}")
    elif taus == "log10":
        maxn = np.log10(len(data))
        taus = (1.0/rate)*np.logspace(0, maxn, int(10*maxn), base=10.0)
        if v:
            print("tau_generator: maxn %.1f" % maxn)
            print("tau_generator: taus=" % taus)
    elif taus == "decade":  # 1, 2, 4, 10, 20, 40, spacing similar to Stable32
        maxn = np.floor(np.log10(len(data)))
        taus = []
        for k in range(int(maxn+1)):
            taus.append(1.0*(1.0/rate)*pow(10.0, k))
            taus.append(2.0*(1.0/rate)*pow(10.0, k))
            taus.append(4.0*(1.0/rate)*pow(10.0, k))

    data, taus = np.array(data), np.array(taus)
    rate = float(rate)
    m = []  # integer averaging factor. tau = m*tau0

    if dev == "adev":
        stop_ratio = 5
    elif dev == "mdev":
        stop_ratio = 4
    elif dev == "tdev":
        stop_ratio = 4
    elif dev == "oadev":
        stop_ratio = 4

    if maximum_m == -1:  # if no limit given
        maximum_m = len(data)/stop_ratio
    # FIXME: should we use a "stop-ratio" like Stable32
    # found in Table III, page 9 of
    # "Evolution of frequency stability analysis software"
    # max(AF) = len(phase)/stop_ratio, where
    # function  stop_ratio
    # adev      5
    # oadev     4
    # mdev      4
    # tdev      4
    # hdev      5
    # ohdev     4
    # totdev    2
    # tierms    4
    # htotdev   3
    # mtie      2
    # theo1     1
    # theoH     1
    # mtotdev   2
    # ttotdev   2
    
    # m = np.round(taus * rate)
    # m = (taus * rate).astype(int)
    m = np.round(taus * rate).astype(int)
    # Debug: Check the calculated taus and m
    # st.write(f"Calculated taus: {taus}")
    # st.write(f"Calculated m: {m}")

    taus_valid1 = m < len(data)
    taus_valid2 = m > 0
    taus_valid3 = m <= maximum_m
    taus_valid = taus_valid1 & taus_valid2 & taus_valid3
    m = m[taus_valid]
    m = m[m != 0]       # m is tau in units of datapoints
    m = np.unique(m)    # remove duplicates and sort

    if v:
        print("tau_generator: ", m)

    if len(m) == 0:
        print("Warning: sanity-check on tau failed!")
        print("   len(data)=", len(data), " rate=", rate, "taus= ", taus)

    taus2 = m / float(rate)

    if even:  # used by Theo1
        m_even_mask = ((m % 2) == 0)
        m = m[m_even_mask]
        taus2 = taus2[m_even_mask]

    # st.write(f"taus returned: {taus2}")
    return data, m, taus2


def remove_small_ns(taus, devs, deverrs, ns):
    """ Remove results with small number of samples.

    If n is small (==1), reject the result

    Parameters
    ----------
    taus: array
        List of tau values for which deviation were computed
    devs: array
        List of deviations
    deverrs: array or list of arrays
        List of estimated errors (possibly a list containing two arrays :
        upper and lower values)
    ns: array
        Number of samples for each point

    Returns
    -------
    (taus, devs, deverrs, ns): tuple
        Identical to input, except that values with low ns have been removed.

    """
    ns_big_enough = ns > 1

    o_taus = taus[ns_big_enough]
    o_devs = devs[ns_big_enough]
    o_ns = ns[ns_big_enough]
    if isinstance(deverrs, list):
        assert len(deverrs) < 3
        o_deverrs = [deverrs[0][ns_big_enough], deverrs[1][ns_big_enough]]
    else:
        o_deverrs = deverrs[ns_big_enough]
    if len(o_devs) == 0:
        print("remove_small_ns() nothing remains!?")
        raise UserWarning

    return o_taus, o_devs, o_deverrs, o_ns


def adev(data, rate, data_type, taus=None):
    """ Allan deviation.
        Classic - use only if required - relatively poor confidence.

    .. math::

        \\sigma^2_{ADEV}(\\tau) = { 1 \\over 2 \\tau^2 }
        \\langle ( {x}_{n+2} - 2x_{n+1} + x_{n} )^2 \\rangle
        = { 1 \\over 2 (N-2) \\tau^2 }
        \\sum_{n=1}^{N-2} ( {x}_{n+2} - 2x_{n+1} + x_{n} )^2

    where :math:`x_n` is the time-series of phase observations, spaced
    by the measurement interval :math:`\\tau`, and with length :math:`N`.

    Or alternatively calculated from a time-series of fractional frequency:

    .. math::

        \\sigma^{2}_{ADEV}(\\tau) =  { 1 \\over 2 }
        \\langle ( \\bar{y}_{n+1} - \\bar{y}_n )^2 \\rangle

    where :math:`\\bar{y}_n` is the time-series of fractional frequency
    at averaging time :math:`\\tau`


    Parameters
    ----------
    data: np.array
        Input data. Provide either phase or frequency (fractional,
        adimensional).
    rate: float
        The sampling rate for data, in Hz. Defaults to 1.0
    data_type: {'phase', 'freq'}
        Data type, i.e. phase or frequency. Defaults to "phase".
    taus: np.array
        Array of tau values, in seconds, for which to compute statistic.
        Optionally set taus=["all"|"octave"|"decade"] for automatic
        tau-list generation.

    Returns
    -------
    (taus2, ad, ade, ns): tuple
        Tuple of values
    taus2: np.array
        Tau values for which td computed
    ad: np.array
        Computed adev for each tau value
    ade: np.array
        adev errors
    ns: np.array
        Values of N used in each adev calculation
    """
    
    phase = input_to_phase(data, rate, data_type)

    # st.write(f"Taus input:{taus}")
    # st.write(f"Rate in adev : {rate}")
    (phase, m, taus_used) = tau_generator(phase, rate, "adev", taus)
    # st.write(phase)
    # st.write("Taus")
    # st.write(taus_used)
    ad = np.zeros_like(taus_used)
    ade = np.zeros_like(taus_used)
    adn = np.zeros_like(taus_used)

    # st.write(f"Taus used: {taus_used}")
    

    for idx, mj in enumerate(m):  # loop through each tau value m(j)
        (ad[idx], ade[idx], adn[idx]) = calc_adev_phase(phase, rate, mj, mj)

    return remove_small_ns(taus_used, ad, ade, adn)



# This function is not gap resistant. 
def gap_oadev(data, rate=1.0, data_type="phase", taus=None):
    """ Overlapping Allan deviation.
    General purpose - most widely used - first choice.

    .. math::

        \\sigma^2_{OADEV}(m\\tau_0) = { 1 \\over 2 (m \\tau_0 )^2 (N-2m) }
        \\sum_{n=1}^{N-2m} ( {x}_{n+2m} - 2x_{n+1m} + x_{n} )^2

    where :math:`\\sigma_{OADEV}(m\\tau_0)` is the overlapping Allan
    deviation at an averaging time of :math:`\\tau=m\\tau_0`, and
    :math:`x_n` is the time-series of phase observations, spaced by the
    measurement interval :math:`\\tau_0`, with length :math:`N`.

    Parameters
    ----------
    data: np.array
        Input data. Provide either phase or frequency (fractional,
        adimensional).
    rate: float
        The sampling rate for data, in Hz. Defaults to 1.0
    data_type: {'phase', 'freq'}
        Data type, i.e. phase or frequency. Defaults to "phase".
    taus: np.array
        Array of tau values, in seconds, for which to compute statistic.
        Optionally set taus=["all"|"octave"|"decade"] for automatic
        tau-list generation.

    Returns
    -------
    (taus2, ad, ade, ns): tuple
        Tuple of values
    taus2: np.array
        Tau values for which td computed
    ad: np.array
        Computed oadev for each tau value
    ade: np.array
        oadev errors
    ns: np.array
        Values of N used in each oadev calculation

    """
    phase = input_to_phase(data, rate, data_type)
    (phase, m, taus_used) = tau_generator(phase, rate, "oadev", taus)
    
    ad = np.zeros_like(taus_used)
    ade = np.zeros_like(taus_used)
    adn = np.zeros_like(taus_used)

    for idx, mj in enumerate(m):  # stride=1 for overlapping ADEV
        (ad[idx], ade[idx], adn[idx]) = calc_adev_phase(phase, rate, mj, 1)

    return remove_small_ns(taus_used, ad, ade, adn)



def edf_simple(N, m, alpha):
    """Equivalent degrees of freedom.
    Simple approximate formulae.

    Parameters
    ----------
    N : int
        the number of phase samples
    m : int
        averaging factor, tau = m * tau0
    alpha: int
        exponent of f for the frequency PSD:
        'wp' returns white phase noise.             alpha=+2
        'wf' returns white frequency noise.         alpha= 0
        'fp' returns flicker phase noise.           alpha=+1
        'ff' returns flicker frequency noise.       alpha=-1
        'rf' returns random walk frequency noise.   alpha=-2
        If the input is not recognized, it defaults to idealized, uncorrelated
        noise with (N-1) degrees of freedom.

    Notes
    -----
       See [Stein1985]_

    Returns
    -------
    edf : float
        Equivalent degrees of freedom

    """

    N = float(N)
    m = float(m)
    if alpha in [2, 1, 0, -1, -2]:
        # NIST SP 1065, Table 5
        if alpha == +2:
            edf = (N + 1) * (N - 2*m) / (2 * (N - m))

        if alpha == 0:
            edf = (((3 * (N - 1) / (2 * m)) - (2 * (N - 2) / N)) *
                   ((4*pow(m, 2)) / ((4*pow(m, 2)) + 5)))

        if alpha == 1:
            a = (N - 1)/(2 * m)
            b = (2 * m + 1) * (N - 1) / 4
            edf = np.exp(np.sqrt(np.log(a) * np.log(b)))

        if alpha == -1:
            if m == 1:
                edf = 2 * (N - 2)/(2.3 * N - 4.9)
            if m >= 2:
                edf = 5 * N**2 / (4 * m * (N + (3 * m)))

        if alpha == -2:
            a = (N - 2) / (m * (N - 3)**2)
            b = (N - 1)**2
            c = 3 * m * (N - 1)
            d = 4 * m**2
            edf = a * (b - c + d)

    else:
        edf = (N - 1)
        print("Noise type not recognized."
              " Defaulting to N - 1 degrees of freedom.")

    return edf


#######################
# Confidence Intervals
ONE_SIGMA_CI = scipy.special.erf(1/np.sqrt(2))
#    = 0.68268949213708585


def confidence_interval(dev, edf, ci=ONE_SIGMA_CI):
    """ returns confidence interval (dev_min, dev_max)
        for a given deviation dev, equivalent degrees of freedom edf,
        and degree of confidence ci.

    Parameters
    ----------
    dev: float
        Mean value (e.g. adev) around which we produce the confidence interval
    edf: float
        Equivalent degrees of freedon
    ci: float, defaults to scipy.special.erf(1/math.sqrt(2))
        for 1-sigma standard error set
        ci = scipy.special.erf(1/math.sqrt(2))
            = 0.68268949213708585

    Returns
    -------
    (dev_min, dev_max): (float, float)
        Confidence interval
    """
    ci_l = min(np.abs(ci), np.abs((ci-1))) / 2
    ci_h = 1 - ci_l

    # function from scipy, works OK, but scipy is large and slow to build
    chi2_l = scipy.stats.chi2.ppf(ci_l, edf)
    chi2_h = scipy.stats.chi2.ppf(ci_h, edf)

    variance = dev*dev
    var_l = float(edf) * variance / chi2_h  # NIST SP1065 eqn (45)
    var_h = float(edf) * variance / chi2_l
    return (np.sqrt(var_l), np.sqrt(var_h))



def calc_gradev_phase(data, rate, mj, stride, confidence, noisetype):
    """ see http://www.leapsecond.com/tools/adev_lib.c
        stride = mj for nonoverlapping allan deviation
        stride = 1 for overlapping allan deviation

        see http://en.wikipedia.org/wiki/Allan_variance
             1       1
         s2y(t) = --------- sum [x(i+2) - 2x(i+1) + x(i) ]^2
                  2*tau^2


        ci: float, defaults to scipy.special.erf(1/math.sqrt(2))
        for 1-sigma standard error set
        ci = scipy.special.erf(1/math.sqrt(2))
            = 0.68268949213708585

    """

    d2 = data[2 * int(mj)::int(stride)]
    d1 = data[1 * int(mj)::int(stride)]
    d0 = data[::int(stride)]

    n = min(len(d0), len(d1), len(d2))

    v_arr = d2[:n] - 2 * d1[:n] + d0[:n]

    # only average for non-nans
    n = len(np.where(np.isnan(v_arr) == False)[0])  # noqa

    if n == 0:
        RuntimeWarning("Data array length is too small: %i" % len(data))
        n = 1

    N = len(np.where(np.isnan(data) == False)[0])  # noqa

    # a summation robust to nans
    s = np.nansum(v_arr * v_arr)

    dev = np.sqrt(s / (2.0 * n)) / mj * rate
    # deverr = dev / np.sqrt(n) # old simple errorbars
    if noisetype == 'wp':
        alpha = 2
    elif noisetype == 'wf':
        alpha = 0
    elif noisetype == 'fp':
        alpha = -2
    else:
        alpha = None

    if n > 1:
        edf = edf_simple(N, mj, alpha)
        deverr = confidence_interval(dev, confidence, edf)
    else:
        deverr = [0, 0]

    return dev, deverr, n


# Gap resistant overlapping Allan deviation 

def oadev(data, rate=1.0, data_type="phase", taus=None, ci=0.9, noisetype='wp'):
    """ Gap resistant overlapping Allan deviation

    Parameters
    ----------
    data: np.array
        Input data. Provide either phase or frequency (fractional,
        adimensional). Warning : phase data works better (frequency data is
        first trantformed into phase using numpy.cumsum() function, which can
        lead to poor results).
    rate: float
        The sampling rate for data, in Hz. Defaults to 1.0
    data_type: {'phase', 'freq'}
        Data type, i.e. phase or frequency. Defaults to "phase".
    taus: np.array
        Array of tau values, in seconds, for which to compute statistic.
        Optionally set taus=["all"|"octave"|"decade"] for automatic
        tau-list generation.
    ci: float
        the total confidence interval desired, i.e. if ci = 0.9, the bounds
        will be at 0.05 and 0.95.
    noisetype: string
        the type of noise desired:
        'wp' returns white phase noise.
        'wf' returns white frequency noise.
        'fp' returns flicker phase noise.
        'ff' returns flicker frequency noise.
        'rf' returns random walk frequency noise.
        If the input is not recognized, it defaults to idealized, uncorrelated
        noise with (N-1) degrees of freedom.

    Returns
    -------
    taus: np.array
        list of tau vales in seconds
    adev: np.array
        deviations
    [err_l, err_h] : list of len()==2, np.array
        the upper and lower bounds of the confidence interval taken as
        distances from the the estimated two sample variance.
    ns: np.array
        numper of terms n in the adev estimate.

    """
    if data_type == "freq":
        print("Warning : phase data is preferred as input to gradev()")
    phase = input_to_phase(data, rate, data_type)
    (data, m, taus_used) = tau_generator(phase, rate, "oadev", taus)

    ad = np.zeros_like(taus_used)
    ade_l = np.zeros_like(taus_used)
    ade_h = np.zeros_like(taus_used)
    adn = np.zeros_like(taus_used)

    for idx, mj in enumerate(m):
        (dev, deverr, n) = calc_gradev_phase(data,
                                             rate,
                                             mj,
                                             1,
                                             ci,
                                             noisetype)
        # stride=1 for overlapping ADEV
        ad[idx] = dev
        ade_l[idx] = deverr[0]
        ade_h[idx] = deverr[1]
        adn[idx] = n

    # Note that errors are split in 2 arrays
    return remove_small_ns(taus_used, ad, [ade_l, ade_h], adn)




# Main Streamlit app
def main():

    # # Initialize or reset session state on first load or as needed
    # if 'data_type' not in st.session_state:
    #     initialize_state()

    # if 'input_data' not in st.session_state or 'data_type' not in st.session_ state:
    #     initialize_state()  # Ensure everything is initialized
    if 'data_type' not in st.session_state or 'input_data' not in st.session_state:
        initialize_state()


    # run_once = 0
    # Create a 6-column layout
    container1 = st.container(border=True)
    container1.subheader("Input configuration ")

    with container1:

        # col1, col2, col3, col4, col5, col6 = st.columns(6)
        # normal_width = (100 - 5.25) / 5  # Subtract 5.25% instead of 5 to balance the extra 5% for col5
        # col5_width = normal_width + 5.25

        # # Convert these into a list of proportions
        # widths = [normal_width, normal_width, normal_width, normal_width, col5_width, normal_width]
        # Create 6 columns with equal width
        col1, col2, col3, col4, col5, col6 = st.columns([1, 1, 1, 1, 1.1, 1])  # Each number represents an equal share of the width

        # col1, col2, col3, col4, col5, col6 = st.columns(widths)
            # Data types available for selection
        data_types = ['Phase/Time Data', 'Fractional Frequency']
        
        # First column for file upload inside a popover
        with col1:
            # st.write("**Upload your files** 👇")
            with st.popover("Upload Data Files 📁", help="Click here to upload files for processing"):
                with st.form("file_upload_form", clear_on_submit=True):
                    files_uploaded = st.file_uploader("**Upload the clock data files**", accept_multiple_files=True)
                    submitted1 = st.form_submit_button("Submit")
                    if submitted1 and files_uploaded:
                        st.session_state.files_uploaded = files_uploaded
                        # Extract filenames and store them in a list
                        filenames = [file.name for file in files_uploaded]
                        # Append the filenames
                        for filename in filenames:
                            st.session_state.valid_filenames.append(filename)
                        if st.session_state.valid_filenames != 0 :
                            st.write(f"Files uploaded: {st.session_state.valid_filenames}")
                            st.success("Files successfully uploaded.")
                            st.session_state.input_data = {'files_uploaded': st.session_state.valid_filenames}

        with col2:
            # st.write("**Select the type of Data** 👇")
            # Popover for Time Data settings
            with st.popover("Input Data Settings :alarm_clock:", help="Choose the data you are trying to process"):
                # st.session_state.data_type = st.radio("Select the type of Data you have:", ['Time Data', 'Frequency Data'])

                # data_type = ['Time Data', 'Frequency Data']
                # st.session_state.data_type = st.radio("Select the type of Data you have:", data_type, index=0 if st.session_state.data_type not in data_type else data_type.index(st.session_state.data_type))
                

                st.session_state.data_type = st.radio(
                "Select the type of Data you have:",
                data_types,
                index=0 if st.session_state.data_type not in data_types else data_types.index(st.session_state.data_type))
            
                if st.session_state.data_type == 'Phase/Time Data':
                    st.session_state.order_of_data = st.selectbox("**Select the units of your data (s)**",
                                            ('s','ns','ps','ms', 'µs'),
                                            help="Choose the unit that best describes your data's time resolution.")
                    st.session_state.freq_scale = 0 # If it time data, this is used just as a flag indicator
                    st.session_state.input_data.update({
                    'data_type': st.session_state.data_type,
                    'units_data': st.session_state.order_of_data,
                    'signal_frequency': st.session_state.freq_scale })
            # # Popover for Frequency Data settings
            # with st.popover("Frequency Data Settings :radio:", help="Configure settings for frequency data processing"):
                # if st.session_state.data_type == 'Frequency Data':
                #     st.session_state.freq_scale = st.select_slider("**Select the frequency of the signal (MHz)**",
                #                                 options=[0, 5, 10, 100],
                #                                 help="Select the base frequency scale in MHz.")
                #     st.session_state.order_of_data = st.selectbox("**Select the units of your data (Hz)**",
                #                                     ('mHz', 'Hz','µHz','nHz'),
                #                                     help="Adjust the frequency scale by specifying the power of ten.")
                #     st.session_state.input_data.update({
                #     'data_type': st.session_state.data_type,
                #     'units_data': st.session_state.order_of_data,
                #     'signal_frequency': st.session_state.freq_scale })

                if st.session_state.data_type == 'Fractional Frequency':
                    st.session_state.freq_scale = st.select_slider("**Select the frequency of the signal (MHz)**",
                                                options=[5, 10, 100],
                                                help="Select the base frequency scale in MHz.")
                    st.session_state.order_of_data = 'Unitless'
                    st.session_state.input_data.update({
                    'data_type': st.session_state.data_type,
                    'units_data': st.session_state.order_of_data,
                    'signal_frequency': st.session_state.freq_scale })
                    
                    # st.warning("Not Operational yet")
        
        # File combination 
        with col3:
            # st.write("**Select the file combination** 👇")
            with st.popover("File Description  :link:", help="Choose how you are combining your files"):
                st.session_state.file_combo = st.radio("Select the file combination:", ['Each file is a different clock','Multiple files of same clock'],help="Choose the file arrangement you have")
                st.session_state.input_data.update({'file_combo': st.session_state.file_combo})
        # Data Format 
        with col4:
            # st.write("**Data Format** 👇")
            with st.popover("Data Format :gear:", help="Select the timestamp and the data columns in your data"):
                timestamp_help_text = (
                    "If your data does not have a timestamp column, choose 'NA'. "
                    "Acceptable Timestamp formats include: "
                    "- MJD (e.g., 56032, 56032.1234); ")
                
                value_help_text = "Data should be numerical or exponential (e.g., 3.14, 2.17e-5)."
                
                st.session_state.timestamp_col = st.selectbox("**Select the TIMESTAMP column**", options=[1,'NA', 2, 3, 4, 5, 6, 7, 8, 9, 10], help=timestamp_help_text)
                st.session_state.data_col = st.selectbox("**Select the DATA column**", options=[2, 3, 1, 4, 5, 6, 7, 8, 9, 10], help=value_help_text)
                st.session_state.input_data.update({'timestamp_col': st.session_state.timestamp_col})
                st.session_state.input_data.update({'data_col': st.session_state.data_col})
        #  
        with col5:
            # st.write("**Measurement Interval**")
            with st.popover("Measurement Interval :straight_ruler:", help="Select the measurement interval in seconds. If the required option is not available in the menu enter it manually selecting OTHER"):
                Meas_int_help_text = "Measurement interval in seconds. You can choose OTHER to enter manually" 
                st.session_state.tau0 = st.selectbox("**Measurement Interval [s]**", options=[1, 10, 100, 3600, 86400, 'other'], help=Meas_int_help_text)
                
                if st.session_state.tau0 == 'other':
                    st.session_state.tau0 = st.number_input("Measurement Interval [s]")
                
                st.session_state.input_data.update({'tau0': st.session_state.tau0})

        with col6:
            # st.write("**Process/Analyse**")
            # proceed = st.button("Plot_data")
            if st.button("Proceed to Data Checking"):
                st.session_state.proceed = True



    # Second container 
    # container2 = st.container(border=True)
    # with container2:
    # col1, sp1, col2, sp2, col3 = st.columns([2,0.1,12,0.1,2])  # Adjust the ratio as needed
    col1, sp1, col2, = st.columns([2,0.1,12])  # Adjust the ratio as needed
    with col1:
        container2 = st.container(border=True)
        # container2.subheader("Overview of the input settings")
        # container2.write("**Overview of the input settings**")
        container2.markdown("**Overview of the input settings**")
        
        # Check if the proceed flag has been set before updating the display
        if st.session_state.proceed or st.session_state.data_loaded:
            
            # Use current session state data
            data_dict = {
                'Files uploaded': st.session_state.input_data['files_uploaded'],
                'Type of Data': st.session_state.input_data['data_type'],
                'Units of data': st.session_state.input_data['units_data'],
                'Signal frequency': f"{st.session_state.input_data['signal_frequency']} MHz",
                'File combo': st.session_state.input_data['file_combo'],
                'Timestamp column': st.session_state.input_data['timestamp_col'],
                'Data column': st.session_state.input_data['data_col'],
                'Measurement Interval (s)': st.session_state.input_data['tau0']
            }
        else:
            # Use placeholders or NA values if the proceed flag has not been set
            data_dict = {key.replace('_', ' ').capitalize(): 'NA' for key in st.session_state.input_data.keys()}

        # Create a DataFrame for display
        # data_df = pd.DataFrame([data_dict]).T  # Creating a DataFrame from a single row dictionary
        for key, value in data_dict.items():
            container2.markdown(f"**{key}:** {value}")

    with col2:
        container3 = st.container(border=True)
        
        # tab1, tab2, tab3 = st.tabs(["Raw Data", "Analyse", "Out come"])
        # The order of the functinality shall be raw_data -> data_range -> detrend -> offset -> outlier -> smoothing -> stability.
        st.session_state.selected = option_menu(
            menu_title = None,
            options = ["Raw Data", "Data Range","Detrend", "Offset", "Outlier", "Smoothing", "Stability", "Out come"],
            icons=["database-fill-up", "arrows-collapse-vertical", "alt","graph-up", "activity", "boxes", "check-circle"],
            default_index =0,
            orientation= "horizontal",
            styles={
            "container": {"padding": "0!important", "background-color": "#fafafa"},
            "icon": {"color": "orange", "font-size": "25px"}, 
            "nav-link": {"font-size": "15px", "text-align": "centre", "margin":"0px", "--hover-color": "#eee"},
            "nav-link-selected": {"background-color": "#2d3142"},}
        )
        
        # Initialize session state
        if 'clk_data_full' not in st.session_state:
            # st.session_state['clk_data_full'] = df
            # st.session_state['filtered_data'] = df
            st.session_state['is_time_type'] = False
        
        # Example usage for updating stability results
        if 'out_display' not in st.session_state:
            initialize_state()
        
        if 'unitmulty' not in st.session_state:
            st.session_state.unitmulty = 1

        if 'y_title' not in st.session_state:
            st.session_state.y_title = "Phase Data [s]"
        
        if st.session_state.selected == "Raw Data" : # This tab is to show the files and plot the raw data  

            # Initialize or adjust session state for checkbox states
            if 'df_display' not in st.session_state:
                st.session_state.df_display = pd.DataFrame({
                "Files uploaded": pd.Series(dtype='str'),
                "Clock Name": pd.Series(dtype='str'),
                "Sample_data": pd.Series(dtype='str'),
                "Choose Clock": pd.Series(dtype='bool') })
                

            if 'checkbox_states' not in st.session_state:
                st.session_state.checkbox_states = [False] * len(st.session_state.df_display)
            
            # Initialize or adjust session state for checkbox states
            num_rows = len(st.session_state.df_display)
            
            # def handle_checkbox_click(checkbox_value, row_index):
                # Update the checkbox state in st.session_state.checkbox_states
                # st.session_state.checkbox_states[row_index,'Choose Clock'] = checkbox_value
            
            def handle_checkbox_click(checkbox_value, clock_name):
                st.session_state.checkbox_states[clock_name] = checkbox_value

            # st.session_state.checkbox_states = st.session_state.checkbox_states[:num_rows]

            
            # Function to synchronize checkbox states with df_display
            def sync_checkbox_states():
                
                for i, row in st.session_state.df_display.iterrows():
                    clock_name = row['Clock Name']
                    if clock_name in st.session_state.checkbox_states:
                        st.session_state.df_display.at[i, 'Choose Clock'] = st.session_state.checkbox_states[clock_name]
                        # st.write(f"Check box states in sync block: {st.session_state.checkbox_states}")
                    else:
                        st.session_state.df_display.at[i, 'Choose Clock'] = False

            def generate_unique_key(index):
                return f"checkbox_{index}"

            
            # Synchronize checkbox states initially
            sync_checkbox_states()
            # st.write(f"Check box states BEFORE: {st.session_state.checkbox_states}") 
            # Define the column configurations
            column_config = {
                "Files uploaded": st.column_config.TextColumn("Files Uploaded", disabled=True,width= "small"),
                "Clock Name": st.column_config.TextColumn("Clock Name", disabled=True),
                "Sample_data": st.column_config.AreaChartColumn("Preview (Initial 1000 data points)",width= "large"),
                # "Choose Clock": st.column_config.CheckboxColumn(label = "Select Clock",width= "small")
                "Choose Clock": st.column_config.CheckboxColumn(label = "Select the Clocks",width= "small")
            }
            # st.write(f"Check box states middle: {st.session_state.checkbox_states}")

            if st.session_state.proceed or st.session_state.data_loaded:  # update the app if proceed button is pressed or keep the current status live
                if st.session_state.files_uploaded:  # check the condition if the user press proceed without uploading the files
                    st.session_state.data_loaded = True
                    if st.session_state.proceed:  # Process the input data and its configuration only when the proceed button is clicked
                        st.session_state.total_data = process_inputs(st.session_state.files_uploaded)
                        st.session_state.proceed = False  # Change the proceed to make it false to avoid the rerun of the app during the changes in the input settings by the user

                    # If the user selection is single clock with multiple files
                    if st.session_state.total_data and 'combined' in st.session_state.total_data and st.session_state.file_combo == 'Multiple files of same clock':
                        df = st.session_state.total_data["combined"]
                        clock_name_mapping = {}
                        if 'Value' in df.columns:
                            clk_data = df['Value'].iloc[:1000].tolist() if len(df['Value']) > 1000 else df['Value'].tolist()
                            clk_data_json = json.dumps(clk_data)  # Convert list to JSON string
                            st.session_state.df_display = pd.DataFrame({
                                "Files uploaded": [st.session_state.valid_filenames],  # All filenames in one cell
                                "Clock Name": 'Clock 1',
                                "Sample_data": clk_data_json,  # First 1000 data points in one cell
                                "Choose Clock": [False]  # Initialize checkboxes as unchecked
                            }).astype({"Choose Clock": "bool"})
                            
                            clock_name_mapping["Clock 1"] = 'combined'  # Map dynamic name to original name

                        else:
                            st.error("'Value' column not found in the DataFrame.")
                    elif st.session_state.total_data:  # If the result is not none/empty and it also means each file is a different clock
                        files_uploaded = []
                        clock_names = []
                        sample_data = []
                        choose_clock = []  # List to handle checkbox state for each clock
                        clock_name_mapping = {}

                        # Process each file as a different clock
                        for idx, (name, df) in enumerate(st.session_state.total_data.items()):
                            
                            if 'Value' in df.columns:
                                clk_data = df['Value'].iloc[:1000].tolist() if len(df['Value']) > 1000 else df['Value'].tolist()
                                clk_data = normalize_values(clk_data)  # Normalize values here
                                clk_data_json = json.dumps(clk_data)
                                files_uploaded.append(name)
                                clock_names.append(f"Clock {idx + 1}")  # Dynamic clock names
                                clock_name_mapping[f"Clock {idx + 1}"] = name  # Map dynamic name to original name
                                sample_data.append(clk_data_json)
                                choose_clock.append(False)  # Default to unchecked
                            else:
                                st.error(f"'Value' column not found in the DataFrame for {name}.")

                        st.session_state.df_display = pd.DataFrame({
                            "Files uploaded": files_uploaded,
                            "Clock Name": clock_names,
                            "Sample_data": sample_data,
                            "Choose Clock": choose_clock  # Checkbox column data
                        }).astype({"Choose Clock": "bool"})
                        
                    
                    # Initialize checkbox states (optional, you might handle this in data processing)
                    if 'checkbox_states' not in st.session_state:
                        st.session_state.checkbox_states = [False] * len(st.session_state.df_display)

                    # Resynchronize checkbox states after updating df_display
                    sync_checkbox_states()

                else:
                    st.error("Please upload the files", icon="⚠️")
            
            # Update the data editor display with the new data
            edited_df = st.data_editor(
                st.session_state.df_display,
                column_config=column_config,
                height=300,
                use_container_width=True,
                hide_index=True,
                num_rows="fixed",
                disabled=["Files uploaded", "Sample_data"]
            )
            
            # Update checkbox states based on the user interactions within st.data_editor
            for i in range(len(edited_df)):
                clock_name = edited_df.at[i, 'Clock Name']
                st.session_state.checkbox_states[clock_name] = edited_df.at[i, 'Choose Clock']
            
            # Sync again after user interaction
            st.session_state.df_display = edited_df
            sync_checkbox_states()
            # st.write(f"Check box states AFTER: {st.session_state.checkbox_states}")

            if st.button("Process Selected Clocks"):
                # Extract selected rows based on the 'Choose Clock' column
                selected_data = st.session_state.df_display[st.session_state.df_display['Choose Clock'] == True]
                
                if selected_data.empty:
                    st.error("Please select at least one clock to process")
                else:
                    st.session_state.clk_sel = True
                    
                    # Collect the names of the selected clocks
                    selected_clock_names = []
                    # Ensure 'Files uploaded' is always treated as a list and process each selected clock
                    for index, row in selected_data.iterrows():
                        clock_name = row["Clock Name"]
                        selected_clock_names.append(clock_name)  # Add clock name to the list
                        uploaded_files = row["Files uploaded"]
                        if isinstance(uploaded_files, str):
                            uploaded_files = [uploaded_files]

                        for file in uploaded_files:
                            # st.write(f"Selected Clock file for further processing: {file}")  # Print each selected file name
                            # st.write(f"Selected {clock_name} for further processing")
                            file_key = file.split('.')[0]

                            if st.session_state.file_combo == 'Multiple files of same clock':
                                if 'combined' in st.session_state.total_data:
                                    st.session_state.clk_to_analyse = st.session_state.total_data['combined']
                                    st.session_state.clk_filename = file
                                else:
                                    st.error("No combined data available")
                            else:
                                if file_key in st.session_state.total_data:
                                    st.session_state.clk_to_analyse = st.session_state.total_data[file_key]
                                    st.session_state.clk_filename = file
                                else:
                                    st.error(f"No data available for {file_key}")
                    # Print the names of the selected clocks in a single statement
                    st.write(f"Selected Clocks for processing: {', '.join(selected_clock_names)}")
                    # st.write(st.session_state.clk_to_analyse)            
                    # st.write(st.session_state.total_data)        
                    

                    # Initialize the data frames for all the tabs for the selected clocks
                    for clock_name in selected_clock_names:
                        initialize_clock_data(clock_name)
                        original_key = clock_name_mapping[clock_name]  # Get the original key
                        st.session_state.data[clock_name]['raw_data'] = st.session_state.total_data[original_key].copy()
                        st.session_state.data[clock_name]['data_range'] = st.session_state.total_data[original_key].copy()
                        st.session_state.data[clock_name]['detrend'] = st.session_state.total_data[original_key].copy()
                        st.session_state.data[clock_name]['offset'] = st.session_state.total_data[original_key].copy()
                        st.session_state.data[clock_name]['outlier'] = st.session_state.total_data[original_key].copy()
                        st.session_state.data[clock_name]['smoothing'] = st.session_state.total_data[original_key].copy()
                        st.session_state.data[clock_name]['stability'] = st.session_state.total_data[original_key].copy()
                        st.session_state.data[clock_name]['outcome'] = st.session_state.total_data[original_key].copy()

        if st.session_state.selected == "Data Range":
            if 'clk_sel' in st.session_state and st.session_state.clk_sel:
                selected_clock_names = st.session_state.df_display[st.session_state.df_display['Choose Clock'] == True]["Clock Name"].tolist()
                selected_clock_names.append("Combine Clocks")

                selected_clock = st.radio(":blue-background[**Select Clock for Range Selection**]", selected_clock_names, horizontal=True)

                if selected_clock == "Combine Clocks":
                    combined_data = []
                    combined_info = []  # To store clock info for CSV header
                    for index, row in st.session_state.df_display.iterrows():
                        if row["Clock Name"] in selected_clock_names[:-1]:
                            clock_name = row["Clock Name"]
                            clk_analysis = get_latest_data(clock_name, 'data_range')
                            if clock_name in st.session_state.clock_ranges:
                                start_range = st.session_state.clock_ranges[clock_name]['start_range']
                                end_range = st.session_state.clock_ranges[clock_name]['end_range']
                                filtered_data = clk_analysis[(clk_analysis["Timestamp"] >= start_range) & (clk_analysis["Timestamp"] <= end_range)]
                                combined_data.append((filtered_data, clock_name))
                                update_action(clock_name, 'Data Range', f"Start: {start_range}, End: {end_range}")
                            else:
                                combined_data.append((clk_analysis, clock_name))

                            # Collect info for CSV header
                            combined_info.append({
                                'clock_name': clock_name,
                                'start_range': start_range,
                                'end_range': end_range
                            })

                    fig = go.Figure()
                    for data, name in combined_data:
                        fig.add_trace(go.Scatter(x=data["Timestamp"], y=data["Value"], mode='markers', name=name))

                    fig.update_xaxes(tickformat=".2f")
                    fig.update_layout(
                        title="Data of Clock",
                        xaxis_title="MJD",
                        yaxis_title=st.session_state.y_title,
                        yaxis=dict(tickmode='auto', nticks=10),
                        showlegend=True,
                        xaxis=dict(tickformat=".1f", tickfont=dict(size=14, color="black"), exponentformat='none'),
                        height=600
                    )

                    st.plotly_chart(fig, use_container_width=True)

                    # Create a DataFrame for combined data
                    combined_df = pd.DataFrame()
                    for data, name in combined_data:
                        temp_df = data[['Timestamp', 'Value']].copy()
                        temp_df.columns = [f'Timestamp_{name}', f'Range_Selected_{name}']
                        combined_df = pd.concat([combined_df, temp_df], axis=1)

                    # Create CSV header with clock info
                    csv_header = "# Combined Clock Data\n"
                    for info in combined_info:
                        csv_header += f"# Clock Name: {info['clock_name']}, Start Range: {info['start_range']}, End Range: {info['end_range']}\n"
                    csv_header += "# Data\n"

                    # Convert DataFrame to CSV format
                    # Convert the columns data to proper scientific format 
                    value_columns = [col for col in combined_df.columns if 'Range_Selected' in col]
                    
                    # Apply the formatting function only to the "Value" columns
                    combined_df[value_columns] = combined_df[value_columns].applymap(format_scientific2)

                    csv_data = csv_header + combined_df.to_csv(index=False, lineterminator='\n')

                    # Add download button
                    st.download_button(
                        label="Download Combined Data as CSV",
                        data=csv_data,
                        file_name='combined_data_range.csv',
                        mime='text/csv',
                    )

                else:  # not the combined clocks
                    for index, row in st.session_state.df_display.iterrows():
                        if row["Clock Name"] == selected_clock:
                            clock_name = row["Clock Name"]
                            # st.write(f"Clock  Name: {clock_name}")
                            clk_analysis = get_latest_data(clock_name, 'data_range')
                            # st.write(clk_analysis)
                            st.session_state['clk_data_full'] = clk_analysis.copy()

                            if clock_name not in st.session_state.clock_ranges:
                                st.session_state.clock_ranges[clock_name] = {
                                    'start_range': clk_analysis["Timestamp"].iloc[0],
                                    'end_range': clk_analysis["Timestamp"].max()
                                }

                            start_range = st.session_state.clock_ranges[clock_name]['start_range']
                            end_range = st.session_state.clock_ranges[clock_name]['end_range']

                            if f"start_float_{clock_name}" not in st.session_state:
                                st.session_state[f"start_float_{clock_name}"] = str(start_range)
                            if f"end_float_{clock_name}" not in st.session_state:
                                st.session_state[f"end_float_{clock_name}"] = str(end_range)

                            reset_clicked = st.button("Reset to full Data", key=f"reset_button_{clock_name}")

                            if reset_clicked:
                                full_data = st.session_state.data[clock_name]['raw_data']
                                st.session_state.clock_ranges[clock_name] = {
                                    'start_range': full_data["Timestamp"].iloc[0],
                                    'end_range': full_data["Timestamp"].max()
                                }
                                start_range = full_data["Timestamp"].iloc[0]
                                end_range = full_data["Timestamp"].max()

                                # Directly update session state values
                                st.session_state[f"start_float_{clock_name}"] = str(start_range)
                                st.session_state[f"end_float_{clock_name}"] = str(end_range)

                                st.session_state['filtered_data'] = full_data.copy()
                                st.session_state.data[clock_name]['data_range'] = full_data.copy()

                                col1, col2, col3, col4, col5 = st.columns(5)

                                # Use session state values to initialize the input fields
                                start_range_input = col1.text_input(
                                    "Horizontal axis start", value=st.session_state[f"start_float_{clock_name}"], key=f"start_{clock_name}"
                                )
                                end_range_input = col3.text_input(
                                    "Horizontal axis end", value=st.session_state[f"end_float_{clock_name}"], key=f"end_{clock_name}"
                                )

                                fig = create_plots(st.session_state['filtered_data']['Timestamp'], st.session_state['filtered_data']['Value'])
                                st.plotly_chart(fig, use_container_width=True)

                            else:
                                col1, col2, col3, col4, col5 = st.columns(5)

                                # Use session state values to initialize the input fields
                                start_range_input = col1.text_input(
                                    "Horizontal axis start", value=st.session_state[f"start_float_{clock_name}"], key=f"start_{clock_name}", on_change=lambda: update_range(clock_name, "start", st.session_state[f"start_{clock_name}"])
                                )
                                
                                end_range_input = col3.text_input(
                                    "Horizontal axis end", value=st.session_state[f"end_float_{clock_name}"], key=f"end_{clock_name}", on_change=lambda: update_range(clock_name, "end", st.session_state[f"end_{clock_name}"])
                                )

                                try:
                                    if start_range_input and end_range_input:
                                        start_range = float(start_range_input)
                                        end_range = float(end_range_input)

                                        if start_range < end_range:
                                            st.session_state.clock_ranges[clock_name]['start_range'] = start_range
                                            st.session_state.clock_ranges[clock_name]['end_range'] = end_range
                                            st.session_state['filtered_data'] = clk_analysis[(clk_analysis["Timestamp"] >= start_range) & (clk_analysis["Timestamp"] <= end_range)].copy()
                                            # st.session_state.data[clock_name]['data_range'] = clk_analysis[(clk_analysis["Timestamp"] >= start_range) & (clk_analysis["Timestamp"] <= end_range)].copy()

                                            # Update session state values if valid range is provided
                                            st.session_state[f"start_float_{clock_name}"] = str(start_range)
                                            st.session_state[f"end_float_{clock_name}"] = str(end_range)
                                            update_action(clock_name, 'Data Range', f"Start: {start_range}, End: {end_range}")

                                        else:
                                            st.error("End Range must be greater than Start Range")
                                except ValueError as e:
                                    st.error(f"Invalid input: {e}")
                                except Exception as e:
                                    st.error(f"An error occurred: {e}")

                                fig = create_plots(st.session_state['filtered_data']['Timestamp'], st.session_state['filtered_data']['Value'])
                                st.plotly_chart(fig, use_container_width=True)

                            # Store the processed data for this tab
                            st.session_state.data[clock_name]['data_range'] = st.session_state['filtered_data'].copy()
                            st.session_state.data[clock_name]['detrend'] = st.session_state['filtered_data'].copy()
                            # Store the updated data in the session state for the next tab
                            st.session_state.data[clock_name]['offset'] = st.session_state['filtered_data'].copy()
                            st.session_state.data[clock_name]['outlier'] = st.session_state['filtered_data'].copy()
                            st.session_state.data[clock_name]['smoothing'] = st.session_state['filtered_data'].copy()
                            st.session_state.data[clock_name]['stability'] = st.session_state['filtered_data'].copy()
                            st.session_state.data[clock_name]['outcome'] = st.session_state['filtered_data'].copy()
                            # st.write(f"Clock Name: {clock_name}")
                            # st.write(st.session_state['filtered_data'])
                            break

            else:
                st.warning("Please go to the Raw Data tab and select the clocks you want to analyse and click the button Process Selected Clocks",icon="⚠️")

        if st.session_state.selected == "Detrend":
            if 'clk_sel' in st.session_state and st.session_state.clk_sel:
                selected_clock_names = st.session_state.df_display[st.session_state.df_display['Choose Clock'] == True]["Clock Name"].tolist()
                selected_clock_names.append("Combine Clocks")

                selected_clock = st.radio(":blue-background[**Select Clock for Detrending**]", selected_clock_names, horizontal=True)

                if selected_clock != "Combine Clocks":
                    trend_options = ["None", "Linear", "Quadratic"]
                  
                    # Use a temporary variable to store user selection
                    selected_trend = None

                    if selected_clock in st.session_state.trend_selection:
                        selected_trend = st.session_state.trend_selection[selected_clock]  # Use existing selection if available

                    selected_trend = st.radio(
                        ":green-background[**Select trend to plot the residuals**]",
                        trend_options,
                        key=f"trend_selection_{selected_clock}",  # Unique key based on selected clock
                        index=trend_options.index(selected_trend) if selected_trend is not None else 0,  # Set default index
                        horizontal=True
                    )

                    # Update session state only after confirming selection
                    if selected_trend is not None:
                        st.session_state.trend_selection[selected_clock] = selected_trend
                    
                    if 'filtered_data_dt' not in st.session_state:
                        st.session_state['filtered_data_dt'] = {}
                    # st.session_state.trend_selection[selected_clock] = selected_trend
                    
                    trend_info = []
                    equation = ""  # Initialize the equation variable

                    # Process the data for the selected clock

                    clk_data_detrend = get_latest_data(selected_clock, 'data_range')
                        

                    if selected_trend == 'None':
                        residuals = clk_data_detrend.copy()
                    elif selected_trend == 'Linear' or selected_trend == 'Quadratic':
                        residual_values, slope, coeffs = remove_trend(clk_data_detrend['Value'], selected_trend)
                        residuals = clk_data_detrend.copy()
                        residuals['Value'] = residual_values  # Update residuals DataFrame with the detrended values
                        
                        if selected_trend == 'Linear':
                            equation = f"x(t) = {slope:.2e}t + {coeffs[0]:.2e}"
                            trend_info.append({
                                "Type": "Linear",
                                "Parameter": "Equation",
                                "Value": equation
                            })
                            st.session_state.trend_slopes[selected_clock] = slope
                        elif selected_trend == 'Quadratic':
                            equation = f"x(t) = {coeffs[0]:.2e}t^2 + {coeffs[1]:.2e}t + {coeffs[2]:.2e}"
                            trend_info.append({
                                "Type": "Quadratic",
                                "Parameter": "Equation",
                                "Value": equation
                            })
                            st.session_state.trend_coeffs[selected_clock] = coeffs


                    update_action(selected_clock, 'Detrend', f"Method: {selected_trend}, Equation: {equation}")
                    # Store the updated data in the session state for the current tab
                    st.session_state.data[selected_clock]['detrend'] =  residuals
                    # Store the updated data in the session state for the next tab
                    st.session_state.data[selected_clock]['offset'] =  residuals
                    st.session_state.data[selected_clock]['outlier'] =  residuals
                    st.session_state.data[selected_clock]['smoothing'] =  residuals
                    st.session_state.data[selected_clock]['stability'] =  residuals
                    st.session_state.data[selected_clock]['outcome'] =  residuals

                    # st.write("this is residual")
                    # st.write(residuals)

                    data_to_plot = residuals['Value']
                    timestamps = residuals['Timestamp']
                    # Create plot outside the loop
                    fig = create_plots(timestamps, data_to_plot)
                    st.plotly_chart(fig, use_container_width=True)

                    if trend_info:
                        trend_df = pd.DataFrame(trend_info)
                        st.table(trend_df[['Parameter', 'Value']])


                else:  # If combined clock is selected
                    if 'combined_clocks' not in st.session_state.data:
                        st.session_state.data['combined_clocks'] = {'detrend': pd.DataFrame()}

                    combined_data = []
                    detrend_info = []
                    
                    for index, row in st.session_state.df_display.iterrows():
                        if row["Clock Name"] in selected_clock_names[:-1]:
                            clock_name = row["Clock Name"]
                            clk_analysis = get_latest_data(clock_name, 'detrend')

                            if clock_name in st.session_state.clock_ranges:
                                start_range = st.session_state.clock_ranges[clock_name]['start_range']
                                end_range = st.session_state.clock_ranges[clock_name]['end_range']
                                filtered_data = clk_analysis.loc[(clk_analysis["Timestamp"] >= start_range) & (clk_analysis["Timestamp"] <= end_range)].copy()
                            else:
                                filtered_data = clk_analysis.copy()

                            trend = st.session_state.trend_selection.get(clock_name, "None")
                            residuals, slope, coeffs = remove_trend(filtered_data['Value'].values, trend)
                            filtered_data['Detrended_Value'] = residuals
                            filtered_data['Value'] = residuals  # Update Value with Detrended_Value
                            combined_data.append((filtered_data, clock_name))
                            equation = ""
                            if trend == 'Linear':
                                equation = f"x(t) = {slope:.2e}t + {coeffs[0]:.2e}"
                                st.session_state.trend_slopes[clock_name] = slope
                                st.session_state.trend_intercepts[selected_clock] = coeffs[0]
                            elif trend == 'Quadratic':
                                equation = f"x(t) = {coeffs[0]:.2e}t^2 + {coeffs[1]:.2e}t + {coeffs[2]:.2e}"
                                st.session_state.trend_coeffs[clock_name] = coeffs
                            detrend_info.append({
                                "Clock Name": clock_name,
                                "Trend": trend,
                                "Equation": equation
                            })

                    fig = go.Figure()
                    for data, name in combined_data:
                        fig.add_trace(go.Scatter(x=data["Timestamp"], y=data["Detrended_Value"], mode='markers', name=name))

                    fig.update_xaxes(tickformat=".1f")
                    fig.update_layout(
                        title="Detrended Data of Clocks",
                        xaxis_title="MJD",
                        yaxis_title=st.session_state.y_title,
                        yaxis=dict(tickmode='auto', nticks=10),
                        showlegend=True,
                        xaxis=dict(tickformat=".1f", tickfont=dict(size=14, color="black"), exponentformat='none'),
                        height=600
                    )

                    st.plotly_chart(fig, use_container_width=True)

                    st.markdown("### Detrend Information for Each Clock")

                    if detrend_info:
                        detrend_df = pd.DataFrame(detrend_info)
                        st.table(detrend_df[['Clock Name', 'Trend', 'Equation']])

                    combined_df = pd.DataFrame()
                    combined_info = []  # List to store combined info for header

                    for data, name in combined_data:
                        temp_df = data[['Timestamp', 'Detrended_Value']].copy()
                        temp_df.columns = [f'Timestamp_{name}', f'Detrended_Value_{name}']
                        combined_df = pd.concat([combined_df, temp_df], axis=1)

                        # Collecting combined info for header
                        start_range = st.session_state.clock_ranges.get(name, {}).get('start_range', 'N/A')
                        end_range = st.session_state.clock_ranges.get(name, {}).get('end_range', 'N/A')
                        combined_info.append({
                            "clock_name": name,
                            "start_range": start_range,
                            "end_range": end_range
                        })

                    st.session_state.data['combined_clocks']['detrend'] = combined_df.copy()  # Store combined detrended data

                    # Create CSV header with clock info and detrend information
                    csv_header = "# Combined Clock Data\n"
                    for info in combined_info:
                        csv_header += f"# Clock Name: {info['clock_name']}, Start Range: {info['start_range']}, End Range: {info['end_range']}\n"
                    if detrend_info:
                        for detrend in detrend_info:
                            csv_header += f"# Clock Name: {detrend['Clock Name']}, Trend: {detrend['Trend']}, Equation: {detrend['Equation']}\n"
                    csv_header += "# Data\n"
                    
                    
                    # Convert DataFrame to CSV format
                    # Convert the columns data to proper scientific format 
                    value_columns = [col for col in combined_df.columns if 'Detrended_Value' in col]
                    
                    # Apply the formatting function only to the "Value" columns
                    combined_df[value_columns] = combined_df[value_columns].applymap(format_scientific2)

                    # combined_df = combined_df.round(2)
                    csv_data_detrend = csv_header + combined_df.to_csv(index=False, lineterminator='\n')

                    st.download_button(
                        label="Download Combined Data as CSV",
                        data=csv_data_detrend,
                        file_name='combined_detrended_data.csv',
                        mime='text/csv',
                    )

            else:
                st.warning("Please go to the Raw Data tab and select the clocks you want to analyse and click the button Process Selected Clocks",icon="⚠️")

        if st.session_state.selected == "Offset":
            if 'clk_sel' in st.session_state and st.session_state.clk_sel:
            
                selected_clock_names = st.session_state.df_display[st.session_state.df_display['Choose Clock'] == True]["Clock Name"].tolist()
                selected_clock_names.append("Combine Clocks")

                selected_clock = st.radio(":blue-background[**Select Clock for Offset Removal**]", selected_clock_names, horizontal=True)

                if selected_clock != "Combine Clocks":
                    offset_options = ["None", "Remove Offset[Mean Value]"]

                    if selected_clock not in st.session_state.offset_selection:
                        st.session_state.offset_selection[selected_clock] = "None"

                    selected_offset = st.radio(
                        ":green-background[**Select Offset Option**]",
                        offset_options,
                        key=f"offset_selection_{selected_clock}",  # Unique key based on selected clock
                        index=offset_options.index(st.session_state.offset_selection[selected_clock]),
                        horizontal=True
                    )

                    st.session_state.offset_selection[selected_clock] = selected_offset

                    clock_data = get_latest_data(selected_clock, 'offset').dropna().copy()

                    mean_before_removal = np.mean(clock_data['Value'].values)
                    st.session_state.offset_means_before[selected_clock] = mean_before_removal
                    selected_offset = st.session_state.offset_selection[selected_clock]

                    if selected_offset == "Remove Offset[Mean Value]":
                        offset_removed_data, mean_before_removal = remove_offset(clock_data['Value'].values)
                        clock_data['Offset_Removed_Value'] = offset_removed_data
                        mean_after_removal = np.mean(offset_removed_data)
                        st.session_state.offset_means_after[selected_clock] = mean_after_removal
                    else:
                        clock_data['Offset_Removed_Value'] = clock_data['Value']
                        mean_after_removal = mean_before_removal

                    data_to_plot = clock_data['Offset_Removed_Value'].values
                    timestamps = clock_data['Timestamp']
                    fig = create_plots(timestamps, data_to_plot)
                    st.plotly_chart(fig, use_container_width=True)

                    st.markdown(f"**Mean Value (Before Removal):** {mean_before_removal:.2e} [{st.session_state.order_of_data}]")
                    if selected_offset == "Remove Offset[Mean Value]":
                        st.markdown(f"**Mean Value (After Removal):** {mean_after_removal:.2e} [{st.session_state.order_of_data}]")
                        update_action(selected_clock, 'Offset Removed', f"Method: {selected_offset}, Mean Before: {mean_before_removal:.2e}, Mean After: {mean_after_removal:.2e}")

                    st.session_state.data[selected_clock]['offset'] = clock_data.copy()
                    st.session_state.data[selected_clock]['outlier'] = clock_data.copy()
                    st.session_state.data[selected_clock]['smoothing'] = clock_data.copy()
                    st.session_state.data[selected_clock]['stability'] = clock_data.copy()
                    st.session_state.data[selected_clock]['outcome'] = clock_data.copy()

                else:
                    combined_data = []
                    combined_info = []

                    for index, row in st.session_state.df_display.iterrows():
                        if row["Clock Name"] in selected_clock_names[:-1]:
                            clock_name = row["Clock Name"]
                            clock_data = get_latest_data(clock_name, 'offset').dropna().copy()

                            mean_before_removal = np.mean(clock_data['Value'].values)
                            st.session_state.offset_means_before[clock_name] = mean_before_removal
                            offset_option = st.session_state.offset_selection.get(clock_name, "None")
                            if offset_option == "Remove Offset[Mean Value]":
                                offset_removed_data, mean_before_removal = remove_offset(clock_data['Value'].values)
                                clock_data['Offset_Removed_Value'] = offset_removed_data
                                mean_after_removal = np.mean(offset_removed_data)
                                st.session_state.offset_means_after[clock_name] = mean_after_removal
                            else:
                                clock_data['Offset_Removed_Value'] = clock_data['Value']
                                mean_after_removal = mean_before_removal

                            combined_data.append((clock_data, clock_name))

                            combined_info.append({
                                "Clock Name": clock_name,
                                "Offset Option": offset_option,
                                "Mean Value (Before Removal)": f"{mean_before_removal:.2e} [{st.session_state.order_of_data}]",
                                "Mean Value (After Removal)": f"{mean_after_removal:.2e} [{st.session_state.order_of_data}]" if offset_option == "Remove Offset[Mean Value]" else ""
                            })

                    fig = go.Figure()
                    for data, name in combined_data:
                        fig.add_trace(go.Scatter(x=data["Timestamp"], y=data["Offset_Removed_Value"], mode='markers', name=name))

                    fig.update_xaxes(tickformat=".1f")
                    fig.update_layout(
                        title="Offset Removed Data of Clocks",
                        xaxis_title="MJD",
                        yaxis_title= st.session_state.y_title,
                        yaxis=dict(tickmode='auto', nticks=10),
                        showlegend=True,
                        xaxis=dict(tickformat=".1f", tickfont=dict(size=14, color="black"), exponentformat='none'),
                        height=600
                    )

                    st.plotly_chart(fig, use_container_width=True)

                    st.markdown("### Combined Information for Each Clock")
                    combined_df = pd.DataFrame(combined_info)
                    st.table(combined_df[['Clock Name', 'Offset Option', 'Mean Value (Before Removal)', 'Mean Value (After Removal)']])

                    combined_data_df = pd.DataFrame()
                    for data, name in combined_data:
                        temp_df = data[['Timestamp', 'Offset_Removed_Value']].copy()
                        temp_df.columns = [f'Timestamp_{name}', f'Offset_Removed_Value_{name}']
                        combined_data_df = pd.concat([combined_data_df, temp_df], axis=1)

                    if 'combined_clocks' not in st.session_state.data:
                        st.session_state.data['combined_clocks'] = {}
                    st.session_state.data['combined_clocks']['offset'] = combined_data_df.copy()

                    
                    csv_header = "# Combined Clock Data\n"
                    for info in combined_info:
                        csv_header += f"# Clock Name: {info['Clock Name']}, Offset Option: {info['Offset Option']}, Mean Value (Before Removal): {info['Mean Value (Before Removal)']}"
                        if info['Offset Option'] == "Remove Offset[Mean Value]":
                            csv_header += f", Mean Value (After Removal): {info['Mean Value (After Removal)']}\n"
                        else:
                            csv_header += "\n"
                    csv_header += "# Data\n"

                    # Convert the columns datra to proper scientific format 
                    value_columns = [col for col in combined_data_df.columns if 'Offset_Removed_Value' in col]
                    
                    # Apply the formatting function only to the "Value" columns
                    combined_data_df[value_columns] = combined_data_df[value_columns].applymap(format_scientific2)

                    csv = csv_header + combined_data_df.to_csv(index=False)
                    st.download_button(label="Download Combined Data as CSV", data=csv, file_name='combined_offset_removed_data.csv', mime='text/csv')

                    st.session_state.data['combined_clocks']['outlier'] = combined_data_df.copy()
                    st.session_state.data['combined_clocks']['smoothing'] = combined_data_df.copy()
                    st.session_state.data['combined_clocks']['stability'] = combined_data_df.copy()
                    st.session_state.data['combined_clocks']['outcome'] = combined_data_df.copy()

            else:
                st.warning("Please go to the Raw Data tab and select the clocks you want to analyse and click the button Process Selected Clocks",icon="⚠️")

        if st.session_state.selected == "Outlier":
            if 'clk_sel' in st.session_state and st.session_state.clk_sel:
            
                selected_clock_names = st.session_state.df_display[st.session_state.df_display['Choose Clock'] == True]["Clock Name"].tolist()
                selected_clock_names.append("Combine Clocks")

                selected_clock = st.radio(":blue-background[**Select Clock for Outlier Removal**]", selected_clock_names, horizontal=True)

                if selected_clock != "Combine Clocks":
                    outlier_options = ["None", "Std_Dev Based", "Remove Selected Outliers"]

                    if selected_clock not in st.session_state.outlier_selection:
                        st.session_state.outlier_selection[selected_clock] = "None"
                        st.session_state.std_threshold[selected_clock] = 50.0

                    selected_outlier = st.radio(
                        ":green-background[**Outlier Removal Method**]",
                        outlier_options,
                        key=f"outlier_selection_{selected_clock}",  # Unique key based on selected clock
                        index=outlier_options.index(st.session_state.outlier_selection[selected_clock]),
                        horizontal=True
                    )

                    st.session_state.outlier_selection[selected_clock] = selected_outlier

                    clock_name = selected_clock
                    clock_data = get_latest_data(clock_name, 'offset').dropna().copy()  # Always get the original data from the 'offset' stage

                    if f'outlier_data_{selected_clock}' not in st.session_state:
                        st.session_state[f'outlier_data_{selected_clock}'] = clock_data.copy()

                    initial_data = clock_data  # Use the original data for filtering
                    removed_outliers = []
                    if selected_outlier == 'None':
                        data_to_plot = initial_data['Value']
                        timestamps = initial_data['Timestamp']
                        fig = create_plots(timestamps, data_to_plot)
                        st.plotly_chart(fig, use_container_width=True)

                    elif selected_outlier == 'Std_Dev Based':
                        reset_button_std = st.button("**Reset Std_Dev Filter**", key=f"reset_std_dev_{selected_clock}")

                        if f"std_threshold_{selected_clock}" not in st.session_state:
                            st.session_state[f"std_threshold_{selected_clock}"] = 50.0

                        cole1, cole2, cole3, cole4 = st.columns(4)
                        with cole1:
                            std_threshold = st.number_input(
                                f"Standard Deviation Multiplier for {selected_clock}",
                                min_value=0.1,
                                max_value=100.0,
                                value=float(st.session_state.get(f"std_threshold_{selected_clock}", 50.0)),
                                step=0.1,
                                format="%0.1f",
                                key=f"std_threshold_input_{selected_clock}_{st.session_state.get(f'std_threshold_{selected_clock}')}",
                                help="Minimum threshold limit is 0.1 time of Std Dev and Max threshold limit is 100.0 times of Std Dev in steps of 0.1"
                            )

                        if reset_button_std:
                            std_threshold = 50.0  # Reset to default
                            st.session_state[f"std_threshold_{selected_clock}"] = 50.0 # Reset to default
                            # st.session_state[f'std_threshold_{selected_clock}'] = std_threshold
                            filtered_data, std_dev, new_std_dev = remove_outliers(initial_data, std_threshold, 'Value')
                            st.session_state[f'outlier_data_{selected_clock}'] = filtered_data
                        else:
                            filtered_data, std_dev, new_std_dev = remove_outliers(initial_data, std_threshold, 'Value')
                            st.session_state[f'outlier_data_{selected_clock}'] = filtered_data

                        timestamps = st.session_state[f'outlier_data_{selected_clock}']['Timestamp']
                        data_to_plot = st.session_state[f'outlier_data_{selected_clock}']['Value']
                        fig_filtered = create_plots(timestamps, data_to_plot)
                        st.plotly_chart(fig_filtered, use_container_width=True)

                        st.markdown(f"**Standard Deviation for {selected_clock}:** {std_dev:.2e} [ns]")
                        st.markdown(f"**Max Value:** {np.nanmax(filtered_data['Value']):.2e} [ns]")
                        st.markdown(f"**Min Value:** {np.nanmin(filtered_data['Value']):.2e} [ns]")
                        st.markdown(f"**New Standard Deviation (After Removal):** {new_std_dev:.2e} [ns]")

                        st.session_state.std_threshold[selected_clock] = std_threshold
                        update_action(selected_clock, 'Outlier Filtered', f"Method: {selected_outlier}, Std Dev: {std_threshold}")

                    elif selected_outlier == 'Remove Selected Outliers':
                        filtered_data = st.session_state[f'outlier_data_{selected_clock}']
                        st.markdown(":violet-background[**Zoom-in to the outlier and select it/them using a Box or Lasso selection**]")
                        timestamps = filtered_data['Timestamp']
                        data_to_plot = filtered_data['Value']

                        reset_button = st.button(":yellow-background[**Reset Outlier Removal**]", key=f"reset_outliers_{selected_clock}")
                        if reset_button:
                            st.session_state[f'outlier_data_{selected_clock}'] = clock_data.copy()
                            filtered_data_updated = clock_data.copy()
                            timestamps = filtered_data_updated['Timestamp']
                            data_to_plot = filtered_data_updated['Value']

                            st.session_state[f'outlier_data_{selected_clock}'] = clock_data.copy()
                            fig = create_plots(timestamps, data_to_plot)
                            event_data = st.plotly_chart(fig, key=f"outlier_{selected_clock}", on_select="rerun", selection_mode=('points', 'box', 'lasso'),theme="streamlit")
                            # st.plotly_chart(fig, use_container_width=True)
                        else:
                            # filtered_data_updated = clock_data.copy()
                            # timestamps = filtered_data_updated['Timestamp']
                            # data_to_plot = filtered_data_updated['Value']
                            fig_selection = create_plots(timestamps, data_to_plot)
                            event_data = st.plotly_chart(fig_selection, key=f"outlier_{selected_clock}", on_select="rerun", selection_mode=('points', 'box', 'lasso'),theme="streamlit")

                            if event_data and event_data.selection:
                                selected_points = event_data.selection.get("points", [])

                                if selected_points:
                                    selected_points_indices = [point["point_index"] for point in selected_points if "point_index" in point]

                                    filtered_data_updated = filtered_data.copy()
                                    removed_outliers = filtered_data.loc[filtered_data.index.isin(selected_points_indices), 'Value'].tolist()
                                    filtered_data_updated = filtered_data_updated.drop(index=filtered_data_updated.index[selected_points_indices])
                                    st.session_state[f'outlier_data_{selected_clock}'] = filtered_data_updated
                                    st.success(f"Removed {len(selected_points)} outliers from {selected_clock}")

                                    timestamps = filtered_data_updated['Timestamp']
                                    data_to_plot = filtered_data_updated['Value']
                                    fig_updated = create_plots(timestamps, data_to_plot)
                                    st.plotly_chart(fig_updated, use_container_width=True)

                        update_action(selected_clock, 'Outlier Filtered', f"Method: {selected_outlier}")

                    st.session_state.data[selected_clock]['outlier'] = st.session_state[f'outlier_data_{selected_clock}'].copy()
                    st.session_state.data[selected_clock]['smoothing'] = st.session_state[f'outlier_data_{selected_clock}'].copy()
                    st.session_state.data[selected_clock]['stability'] = st.session_state[f'outlier_data_{selected_clock}'].copy()
                    st.session_state.data[selected_clock]['outcome'] = st.session_state[f'outlier_data_{selected_clock}'].copy()


                    # Prepare the CSV data
                    csv_header = f"# Outlier Removal Data: {selected_clock}\n"
                    start_range = st.session_state.clock_ranges.get(selected_clock, {}).get('start_range', 'N/A')
                    end_range = st.session_state.clock_ranges.get(selected_clock, {}).get('end_range', 'N/A')
                    csv_header += f"# Start Range: {start_range}, End Range: {end_range}\n"
                    csv_header += f"# Outlier Removal Method: {selected_outlier}\n"
                    if selected_outlier == 'Std_Dev Based':
                        csv_header += f"# Standard Deviation Multiplier: {std_threshold}\n"
                    elif selected_outlier == 'Remove Selected Outliers':
                        csv_header += "# Removed Outliers: " + ", ".join(f"{outlier:.2f}" for outlier in removed_outliers) + "\n"
                    csv_header += "\n"

                    # Prepare the CSV data with only Timestamp and renamed Value column
                    formatted_data = st.session_state[f'outlier_data_{selected_clock}'][['Timestamp', 'Value']].copy()
                    # Format the 'Value' column in scientific notation
                    formatted_data['Value'] = formatted_data['Value'].map(lambda x: f"{x:.2e}")
                    formatted_data = formatted_data.rename(columns={'Value': 'Outlier_removed'})
                    csv_data = formatted_data.to_csv(index=False)
                    csv_content = csv_header + csv_data

                    # Add the download button
                    st.download_button(
                        label="Download Outlier Removed Data",
                        data=csv_content,
                        file_name=f"{selected_clock}_outlier_removed_data.csv",
                        mime='text/csv'
                    )

                else:
                    combined_data = []
                    combined_info = []
                    outlier_removal_info = []

                    for index, row in st.session_state.df_display.iterrows():
                        if row["Choose Clock"]:
                            clock_name = row["Clock Name"]
                            clock_data = get_latest_data(clock_name, 'outlier').dropna().copy()

                            if not clock_data.empty:
                                max_value = clock_data['Value'].max()
                                min_value = clock_data['Value'].min()
                                std_dev = clock_data['Value'].std()
                            else:
                                max_value = None
                                min_value = None
                                std_dev = None
                                print(f"Warning: Clock {clock_name} dataframe is empty")

                            combined_info.append({
                                "Clock Name": clock_name,
                                "Max Value": max_value,
                                "Min Value": min_value,
                                "Std Dev": std_dev
                            })

                            clock_data = clock_data.rename(columns={'Timestamp': f'Timestamp_{clock_name}', 'Value': f'Outlier_Removed_Value_{clock_name}'})
                            combined_data.append(clock_data)

                            outlier_method = st.session_state.outlier_selection.get(clock_name, "None")
                            if outlier_method == "Std_Dev Based":
                                std_threshold = st.session_state.std_threshold.get(clock_name, 50.0)
                                outlier_removal_info.append({
                                    "Clock Name": clock_name,
                                    "Outlier Method": outlier_method,
                                    "Std Dev Multiplier": std_threshold
                                })
                            elif outlier_method == "Remove Selected Outliers":
                                removed_outliers = st.session_state[f'outlier_data_{clock_name}'].get('removed_outliers', [])
                                outlier_removal_info.append({
                                    "Clock Name": clock_name,
                                    "Outlier Method": outlier_method,
                                    "Removed Outliers": removed_outliers
                                })


                    if combined_data:
                        combined_df = pd.concat(combined_data, axis=1)

                        fig = go.Figure()
                        for clock_data, row in zip(combined_data, st.session_state.df_display.iterrows()):
                            clock_name = row[1]["Clock Name"]
                            fig.add_trace(go.Scatter(
                                x=clock_data[f'Timestamp_{clock_name}'],
                                y=clock_data[f'Outlier_Removed_Value_{clock_name}'],
                                mode='markers',
                                name=clock_name
                            ))

                        fig.update_xaxes(tickformat=".1f")
                        fig.update_layout(
                            title="Outlier Removed Data of Clocks",
                            xaxis_title="MJD",
                            yaxis_title=st.session_state.y_title,
                            yaxis=dict(tickmode='auto', nticks=10),
                            showlegend=True,
                            xaxis=dict(tickformat=".1f", tickfont=dict(size=14, color="black"), exponentformat='none'),
                            height=600
                        )

                        st.plotly_chart(fig, use_container_width=True)

                        st.markdown("### Combined Information for Each Clock")
                        combined_info_df = pd.DataFrame(combined_info)
                        # Round the combined_info values to 2 decimal places
                        combined_info_df['Max Value'] = combined_info_df['Max Value'].apply(lambda x: f"{x:.2e}")
                        combined_info_df['Min Value'] = combined_info_df['Min Value'].apply(lambda x: f"{x:.2e}")
                        combined_info_df['Std Dev'] = combined_info_df['Std Dev'].apply(lambda x: f"{x:.2e}")
                        outlier_methods = [st.session_state.outlier_selection.get(clock_name, "None") for clock_name in combined_info_df['Clock Name']]
                        combined_info_df['Outlier Method'] = outlier_methods

                        st.table(combined_info_df[['Clock Name', 'Outlier Method','Max Value', 'Min Value', 'Std Dev']])
                        # Create CSV header with clock info and filtering method
                        csv_header = "# Combined Clock Data\n"
                        for info in combined_info:
                            start_range = st.session_state.clock_ranges.get(info['Clock Name'], {}).get('start_range', 'N/A')
                            end_range = st.session_state.clock_ranges.get(info['Clock Name'], {}).get('end_range', 'N/A')
                            csv_header += f"# Clock Name: {info['Clock Name']}, Start Range: {start_range}, End Range: {end_range}, Max Value: {info['Max Value']:.2e}, Min Value: {info['Min Value']:.2e}, Std Dev: {info['Std Dev']:.2e}\n"
                            # csv_header += f"# Clock Name: {info['Clock Name']}, Max Value: {info['Max Value']:.2e}, Min Value: {info['Min Value']:.2e}, Std Dev: {info['Std Dev']:.2e}\n"
                        
                        for outlier_info in outlier_removal_info:
                            if outlier_info["Outlier Method"] == "Std_Dev Based":
                                csv_header += f"# Clock Name: {outlier_info['Clock Name']}, Outlier Removal Method: {outlier_info['Outlier Method']}, Std Dev Multiplier: {outlier_info['Std Dev Multiplier']}\n"
                            elif outlier_info["Outlier Method"] == "Remove Selected Outliers":
                                removed_outliers = ", ".join(f"{outlier:.2e}" for outlier in outlier_info['Removed Outliers'])
                                csv_header += f"# Clock Name: {outlier_info['Clock Name']}, Outlier Removal Method: {outlier_info['Outlier Method']}, Removed Outliers: {removed_outliers}\n"
                                                    
                        csv_header += "# Data\n"

                        # Keep only the required columns
                        columns_to_keep = []
                        for clock_name in combined_info_df['Clock Name']:
                            columns_to_keep.append(f'Timestamp_{clock_name}')
                            columns_to_keep.append(f'Outlier_Removed_Value_{clock_name}')

                        combined_df = combined_df[columns_to_keep]

                        # Convert the columns datra to proper scientific format 
                        value_columns = [col for col in combined_df.columns if 'Outlier_Removed_Value' in col]
                        
                        # Apply the formatting function only to the "Value" columns
                        combined_df[value_columns] = combined_df[value_columns].applymap(format_scientific2)

                        csv = csv_header + combined_df.to_csv(index=False, lineterminator='\n')
                        st.download_button(
                            label="Download Combined Data as CSV",
                            data=csv,
                            file_name='combined_outlier_removed_data.csv',
                            mime='text/csv',
                        )

                        # st.session_state.data['combined_clocks']['outlier'] = combined_df.copy()

            else:
                st.warning("Please go to the Raw Data tab and select the clocks you want to analyse and click the button Process Selected Clocks",icon="⚠️")

        if st.session_state.selected == "Smoothing":
            
            if 'clk_sel' in st.session_state and st.session_state.clk_sel:

                selected_clock_names = st.session_state.df_display[st.session_state.df_display['Choose Clock'] == True]["Clock Name"].tolist()
                selected_clock_names.append("Combine Clocks")

                selected_clock = st.radio(":blue-background[**Select Clock for smoothing**]", selected_clock_names, horizontal=True)
                
                action_details = []

                if selected_clock != "Combine Clocks":
                    clock_data = get_latest_data(selected_clock, 'outlier').dropna().copy()

                    # smoothing_method = st.radio("Select Smoothing Method", ['None', 'Moving Avg (non overlapping)', 'Moving Avg (overlapping)'], index=0, horizontal=True)
                    smoothing_method = st.radio(
                        ":green-background[**Select Smoothing Method**]",
                        ['None', 'Moving Avg (non overlapping)', 'Moving Avg (overlapping)'],
                        index=0,
                        horizontal=True,
                        key=f"smooth_method_{selected_clock}"  # Unique key based on selected clock
                    )

                    if smoothing_method == 'None':
                        timestamps = clock_data['Timestamp']
                        data_to_plot = clock_data['Value']
                        fig = create_plots(timestamps, data_to_plot)
                        st.plotly_chart(fig, use_container_width=True)

                        st.session_state[f'smoothed_data_{selected_clock}'] = clock_data.copy()
                        st.session_state[f'smoothing_method_{selected_clock}'] = 'None'
                        st.session_state[f'window_size_{selected_clock}'] = "N/A"
                        action_details.append({
                            "Clock": selected_clock,
                            "Selection": "None",
                            "Sampling Window": "N/A",
                            "Mean Value": clock_data['Value'].mean(),
                            "Std Dev": clock_data['Value'].std()
                        })

                    elif smoothing_method == 'Moving Avg (overlapping)':
                        max_window = len(clock_data[clock_data['Value'].notna()]) // 2
                        window_size_from_state = st.session_state.get(f'window_size_{selected_clock}', 10)

                        if isinstance(window_size_from_state, str) and window_size_from_state.isdigit():
                            window_size_from_state = int(window_size_from_state)
                        else:
                            window_size_from_state = 10

                        colme1, colme2, colme3, colme4 = st.columns(4)
                        with colme1:
                            window_size = st.number_input("Sampling Window Size (< half of the length of data)", min_value=1, max_value=max_window, value=window_size_from_state, step=1)

                        st.session_state[f'window_size_{selected_clock}'] = window_size
                        st.session_state[f'smoothing_method_{selected_clock}'] = 'Moving Avg (overlapping)'
                        
                        timestamps = clock_data['Timestamp'].values
                        valid_timestamps, smoothed_data = smoothing_overlap(clock_data['Value'].values, window_size, timestamps)

                        if len(valid_timestamps) != len(smoothed_data):
                            raise ValueError("Length of valid_timestamps and smoothed_data must be the same")

                        fig = go.Figure()
                        fig.add_trace(go.Scatter(x=timestamps, y=clock_data['Value'], mode='markers', name='Raw Data', marker=dict(color='blue')))
                        fig.add_trace(go.Scatter(x=valid_timestamps, y=smoothed_data, mode='lines', name='Moving Average', line=dict(color='red')))
                        fig.update_layout(title=f"Data for {selected_clock} with Moving Average (overlapping)", xaxis_title="Timestamp", yaxis_title="Value")
                        st.plotly_chart(fig, use_container_width=True)

                        smoothed_data_df = pd.DataFrame({'Timestamp': valid_timestamps, 'Value': smoothed_data}).reset_index(drop=True)
                        st.session_state[f'smoothed_data_{selected_clock}'] = smoothed_data_df
                        update_action(selected_clock, 'Smoothed', f"Method: {smoothing_method}, Window Size: {window_size}")

                        action_details.append({
                            "Clock": selected_clock,
                            "Selection": "Moving Average (overlapping)",
                            "Sampling Window": window_size,
                            "Mean Value": smoothed_data.mean(),
                            "Std Dev": smoothed_data.std()
                        })

                    elif smoothing_method == 'Moving Avg (non overlapping)':
                        max_window = len(clock_data[clock_data['Value'].notna()]) // 2
                        window_size_from_state = st.session_state.get(f'window_size_{selected_clock}', 10)

                        if isinstance(window_size_from_state, str) and window_size_from_state.isdigit():
                            window_size_from_state = int(window_size_from_state)
                        else:
                            window_size_from_state = 10

                        colme1, colme2, colme3, colme4 = st.columns(4)
                        with colme1:
                            window_size = st.number_input("Sampling Window Size (< half of the length of data)", min_value=1, max_value=max_window, value=window_size_from_state, step=1)

                        st.session_state[f'window_size_{selected_clock}'] = window_size
                        st.session_state[f'smoothing_method_{selected_clock}'] = 'Moving Avg (non overlapping)'
                        
                        timestamps = clock_data['Timestamp'].values
                        valid_timestamps, smoothed_data = smoothing_nonoverlap(clock_data['Value'].values, window_size, timestamps)

                        if len(valid_timestamps) != len(smoothed_data):
                            raise ValueError("Length of valid_timestamps and smoothed_data must be the same")

                        fig = go.Figure()
                        fig.add_trace(go.Scatter(x=timestamps, y=clock_data['Value'], mode='markers', name='Raw Data', marker=dict(color='blue')))
                        fig.add_trace(go.Scatter(x=valid_timestamps, y=smoothed_data, mode='lines', name='Moving Average', line=dict(color='red')))
                        fig.update_layout(title=f"Data for {selected_clock} with Moving Average (non overlapping)", xaxis_title="Timestamp", yaxis_title="Value")
                        st.plotly_chart(fig, use_container_width=True)

                        smoothed_data_df = pd.DataFrame({'Timestamp': valid_timestamps, 'Value': smoothed_data}).reset_index(drop=True)
                        st.session_state[f'smoothed_data_{selected_clock}'] = smoothed_data_df
                        update_action(selected_clock, 'Smoothed', f"Method: {smoothing_method}, Window Size: {window_size}")

                        action_details.append({
                            "Clock": selected_clock,
                            "Selection": "Moving Avg (non overlapping)",
                            "Sampling Window": window_size,
                            "Mean Value": smoothed_data.mean(),
                            "Std Dev": smoothed_data.std()
                        })

                    st.session_state.data[selected_clock]['smoothing'] = st.session_state[f'smoothed_data_{selected_clock}'].copy()
                    st.session_state.data[selected_clock]['stability'] = st.session_state[f'smoothed_data_{selected_clock}'].copy()
                    st.session_state.data[selected_clock]['outcome'] = st.session_state[f'smoothed_data_{selected_clock}'].copy()

                else:
                    combined_data = []
                    combined_info = []

                    for index, row in st.session_state.df_display.iterrows():
                        if row["Clock Name"] in selected_clock_names[:-1]:
                            clock_name = row["Clock Name"]
                            clock_data = get_latest_data(clock_name, 'smoothing').dropna().copy()

                            if not clock_data.empty:
                                st.session_state[f'smoothed_data_{clock_name}'] = clock_data.copy()
                                combined_data.append((clock_data, clock_name))

                                max_value = clock_data['Value'].max()
                                min_value = clock_data['Value'].min()
                                std_dev = clock_data['Value'].std()

                                combined_info.append({
                                    "Clock Name": clock_name,
                                    "Max Value": max_value,
                                    "Min Value": min_value,
                                    "Std Dev": std_dev
                                })

                    if combined_data:
                        combined_df_list = []
                        action_details = []

                        for data, name in combined_data:
                            data = data.rename(columns={'Value': f'Value_{name}', 'Timestamp': f'Timestamp_{name}'})
                            combined_df_list.append(data)

                        combined_df = pd.concat(combined_df_list, axis=1)
                        
                        # Drop 'Offset_Removed_Value' columns
                        columns_to_drop = [col for col in combined_df.columns if 'Offset_Removed_Value' in col]
                        combined_df.drop(columns=columns_to_drop, inplace=True)

                        fig_combined = go.Figure()
                        for data, name in combined_data:
                            timestamp_key = f'Timestamp_{name}'
                            value_key = f'Value_{name}'
                            if timestamp_key in combined_df and value_key in combined_df:
                                fig_combined.add_trace(go.Scatter(x=combined_df[timestamp_key], y=combined_df[value_key], mode='markers', name=name))

                        fig_combined.update_layout(title="Combined Data", xaxis_title="MJD", yaxis_title=st.session_state.y_title, height=600)
                        st.plotly_chart(fig_combined, use_container_width=True)

                        for name in [name for _, name in combined_data]:
                            # method = st.session_state.smoothing_method.get(name, "None")
                            # window_size = "N/A" if method == "None" else st.window_size.get(name, "N/A")
                            method = st.session_state.get(f'smoothing_method_{name}', "None")
                            window_size = st.session_state.get(f'window_size_{name}', "N/A")

                            mean_value = combined_df[f'Value_{name}'].mean()
                            std_dev_value = combined_df[f'Value_{name}'].std()
                            
                            action_details.append({
                                "Clock": name,
                                "Selection": method,
                                "Sampling Window": window_size,
                                "Mean Value": f"{mean_value:.2e}" if pd.notnull(mean_value) else "N/A",
                                "Std Dev": f"{std_dev_value:.2e}" if pd.notnull(std_dev_value) else "N/A"
                            })

                        action_df = pd.DataFrame(action_details)
                        action_df['Sampling Window'] = action_df['Sampling Window'].astype(str)

                        # Display the action details table
                        st.markdown("### Combined Information for Each Clock")
                        st.table(action_df[['Clock', 'Selection', 'Sampling Window', 'Mean Value', 'Std Dev']])

                        # Create CSV header
                        csv_header = "# Combined Clock Data\n"
                        for action in action_details:
                            csv_header += f"# Clock: {action['Clock']}, Selection: {action['Selection']}, Sampling Window: {action['Sampling Window']}, Mean Value: {action['Mean Value']}, Std Dev: {action['Std Dev']}\n"
                            # csv_header += f"# Clock: {action['Clock']}, Selection: {action['Selection']}, Sampling Window: {action['Sampling Window']}, Mean Value: {action['Mean Value']:.2e}, Std Dev: {action['Std Dev']:.2e}\n"
                        csv_header += "# Data\n"

                        # Convert the columns data to proper scientific format
                        value_columns = [col for col in combined_df.columns if 'Value_' in col]

                        # # Debug: Print the value columns to be formatted
                        # print("Value columns to be formatted:", value_columns)

                        # Apply the formatting function only to the "Value" columns
                        combined_df[value_columns] = combined_df[value_columns].applymap(format_scientific2)

                        csv = csv_header + combined_df.to_csv(index=False)

                        # Add download button
                        st.download_button(label="Download Combined Data as CSV", data=csv, file_name='combined_data.csv', mime='text/csv')

                        st.smoothed_data = combined_df
                        # st.markdown("### Action Details for Each Clock")
                        # st.table(action_df)

            else:
                st.warning("Please go to the Raw Data tab and select the clocks you want to analyse and click the button Process Selected Clocks",icon="⚠️")

        if st.session_state.selected == "Stability": 
            if 'clk_sel' in st.session_state and st.session_state.clk_sel:
                 
                selected_clk_names = st.session_state.df_display[st.session_state.df_display['Choose Clock'] == True]["Clock Name"].tolist()
                
                st.markdown("""
                    <div style='text-align: center;'>
                        <h3>Stability Analysis</h3>
                        <p><em>[Courtesy: Allan Tools (Note: Computes stability for equally spaced data only)]</em></p>
                    </div>
                    """, unsafe_allow_html=True)
                
                col1, col2 = st.columns(2)

                # Default selections if not set in session state
                if 'analysis_selection' not in st.session_state:
                    st.session_state.analysis_selection = ["ADEV"]
                if 'default_clk' not in st.session_state:
                    st.session_state.default_clk = [selected_clk_names[0]] if selected_clk_names else []

                with col1:
                    # st.markdown("#### Select Stability Analysis Type")
                    analysis_types = ["ADEV", "MDEV", "OADEV", "TDEV"]
                    analysis_selection = st.multiselect("Choose analysis types:", analysis_types, default=st.session_state.analysis_selection)
                    # Store the selection in session state
                    st.session_state.analysis_selection = analysis_selection

                with col2:
                    # st.markdown("#### Select Clocks:")
                    # keep updated the clock selection
                    st.session_state.default_clk = [selected_clk_names[0]] if selected_clk_names else []

                    selected_clks = st.multiselect("Choose clocks:", selected_clk_names, default=st.session_state.default_clk)
                    # Store the selection in session state
                    st.session_state.selected_clks = selected_clks

                plot_data = {}
                csv_data_dict = {}
                colors = ['orange', 'blue', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
                markers = ['o', '*', 's', 'D', '^', 'v', '<', '>', 'P', 'X']

                for clock in st.session_state.selected_clks:
                    clock_name = clock

                    # Get the latest data for the selected clock up to the stability tab
                    clock_data = get_latest_data(clock_name, 'stability').dropna().copy()
                    
                    timestamps = clock_data['Timestamp'].values
                    values = clock_data['Value'].values

                    plot_data[clock_name] = (timestamps, values)
                    csv_data_dict[clock_name] = clock_data.copy()
                    csv_data_dict[clock_name]['Value'] *= 1E-9

                for analysis_type in st.session_state.analysis_selection:
                    plt.figure(figsize=(10, 4))
                    all_tau_values = []
                    all_dev_values = []
                    color_idx = 0
                    tau_values_set = False  # New flag to track if tau_values have been set
                    for clock, (timestamps, values) in plot_data.items():
                        color = colors[color_idx % len(colors)]
                        marker = markers[color_idx % len(markers)]
                        if st.session_state.data_type == 'Phase/Time Data':
                            input_data_type = "phase"
                        elif st.session_state.data_type == 'Fractional Frequency':
                            input_data_type = "frequency"

                        if analysis_type == "TDEV":
                            tau_values, dev_values = plot_tdev(values, st.session_state.tau0, input_data_type, label=clock, color=color, marker=marker)
                        elif analysis_type == "MDEV":
                            tau_values, dev_values = plot_mdev(values, st.session_state.tau0, input_data_type, label=clock, color=color, marker=marker)
                        elif analysis_type == "ADEV":
                            tau_values, dev_values = plot_adev(values, st.session_state.tau0, input_data_type, label=clock, color=color, marker=marker)
                        elif analysis_type == "OADEV":
                            tau_values, dev_values = plot_oadev(values, st.session_state.tau0, input_data_type, label=clock, color=color, marker=marker)
                        color_idx += 1

                        if not tau_values_set:
                            all_tau_values.append(tau_values)
                            tau_values_set = True
                        all_dev_values.append(dev_values)

                        # Ensure all_tau_values and all_dev_values are of the same length before updating the session state
                        max_length = max(len(tau_values), len(dev_values))
                        tau_values = np.array(tau_values, dtype=float)
                        tau_values_padded = np.pad(tau_values, (0, max_length - len(tau_values)), constant_values=np.nan)
                        dev_values_padded = np.pad(dev_values, (0, max_length - len(dev_values)), constant_values=np.nan)

                        # Update stability results for the current clock and analysis type
                        update_stability_results(clock, analysis_type, tau_values_padded, dev_values_padded)
                        
                    st.pyplot(plt)

                    # Ensure all_tau_values and all_dev_values are of the same length
                    max_length = max(max(len(tau) for tau in all_tau_values), max(len(dev) for dev in all_dev_values))
                    all_tau_values = [np.pad(tau, (0, max_length - len(tau)), constant_values=np.nan) for tau in all_tau_values]
                    all_dev_values = [np.pad(dev, (0, max_length - len(dev)), constant_values=np.nan) for dev in all_dev_values]

                    # Save plot as image
                    plot_img_path = f'{analysis_type}_plot.png'
                    plt.savefig(plot_img_path)

                    # Prepare CSV data with header information
                    csv_header = f"# Stability Analysis: {analysis_type}\n"
                    for clock in st.session_state.selected_clks:
                        start_range = st.session_state.clock_ranges.get(clock, {}).get('start_range', 'N/A')
                        end_range = st.session_state.clock_ranges.get(clock, {}).get('end_range', 'N/A')
                        csv_header += f"# Clock: {clock}, Start Range: {start_range}, End Range: {end_range}\n"
                    csv_header += "# Tau (s)," + ",".join([f"{clock}" for clock in st.session_state.selected_clks]) + "\n"

                    # Create a final DataFrame where the 'Tau (s)' column is filled correctly
                    final_rows = []

                    # Get the maximum length of all tau_values
                    max_length = max(len(tau) for tau in all_tau_values)

                    # Iterate through the rows up to the maximum length
                    for i in range(max_length):
                        row_data = [all_tau_values[0][i] if i < len(all_tau_values[0]) else '']
                        for dev_values in all_dev_values:
                            if i < len(dev_values):
                                row_data.append(dev_values[i])
                            else:
                                row_data.append('')
                        final_rows.append(row_data)

                    # Flatten the column headers for the final DataFrame
                    columns = ['Tau (s)'] + [f"{clock}" for clock in st.session_state.selected_clks]

                    # Create the final DataFrame
                    final_df = pd.DataFrame(final_rows, columns=columns)

                    
                    # Drop rows with NaN or inf values in 'Tau (s)' before conversion
                    final_df = final_df[final_df['Tau (s)'].notnull() & ~final_df['Tau (s)'].isin([float('inf'), float('-inf')])]
                    # Convert 'Tau (s)' column to integers
                    final_df['Tau (s)'] = final_df['Tau (s)'].astype(int)

                    # Convert other columns to scientific notation with proper rounding
                    for col in final_df.columns[1:]:  # Skip 'Tau (s)' which is the first column
                        final_df[col] = final_df[col].apply(lambda x: f"{x:.2e}" if pd.notnull(x) else "")

                    # Prepare CSV data with the header
                    csv_data = csv_header + final_df.to_csv(index=False,header=False, lineterminator='\n')


                    # Apply custom CSS for sticky header
                    st.markdown(
                        """
                        <style>
                        .stDataFrame thead tr th {
                            position: sticky;
                            top: 0;
                            background: white;
                            z-index: 1;
                        }
                        </style>
                        """,
                        unsafe_allow_html=True,
                    )

                    # Download buttons side by side
                    col1, col2, col3, col4, col5 = st.columns(5)
                    with col2:
                        st.download_button(
                            label=f"Download {analysis_type} Data",
                            data=csv_data,
                            file_name=f'{analysis_type}_data.csv',
                            mime='text/csv'
                        )
                    with col4:
                        with open(plot_img_path, 'rb') as file:
                            st.download_button(
                                label=f"Download {analysis_type} Plot",
                                data=file,
                                file_name=plot_img_path,
                                mime="image/png"
                            )

                    # # Mark the stability analysis as done for each selected clock
                    # for clock in selected_clocks:
                    #     update_action(clock, f'{analysis_type} Analysis', 'Done')

            else:
                st.warning("Please go to the Raw Data tab and select the clocks you want to analyse and click the button Process Selected Clocks",icon="⚠️")

        if st.session_state.selected == "Out come":
            if 'clk_sel' in st.session_state and st.session_state.clk_sel:
                # st.markdown("### Outcome of the Actions")
                
                st.markdown("""
                    <div style='text-align: center;'>
                        <h3>Outcome of the Actions</h3>
                        <p><em>[Display only the actions you performed on the data so far]</em></p>
                    </div>
                    """, unsafe_allow_html=True)

                if 'df_display' in st.session_state:
                    selected_clock_names = st.session_state.df_display[st.session_state.df_display['Choose Clock'] == True]["Clock Name"].tolist()

                    if "Combine Clocks" in selected_clock_names:
                        selected_clock_names.remove("Combine Clocks")

                    outcome_data = []
                    for clock_name in selected_clock_names:
                        # st.write(f"Processing clock: {clock_name}")
                        if clock_name in st.session_state.clock_ranges:
                            start_range = st.session_state.clock_ranges[clock_name]['start_range']
                            end_range = st.session_state.clock_ranges[clock_name]['end_range']
                            update_action(clock_name, 'Data Range', f"Start: {start_range}<br>End: {end_range}")
                        

                        if clock_name in st.session_state.trend_selection:
                            trend = st.session_state.trend_selection[clock_name]
                            equation = "None"
                            if trend == 'Linear':
                                slope = st.session_state.trend_slopes.get(clock_name)
                                intercept = st.session_state.trend_intercepts.get(clock_name)
                                if slope is not None and intercept is not None:
                                    equation = f"x(t) = {slope:.2e}t + {intercept:.2e}"
                            elif trend == 'Quadratic':
                                coeffs = st.session_state.trend_coeffs.get(clock_name)
                                if coeffs is not None and all(c is not None for c in coeffs):
                                    equation = f"x(t) = {coeffs[0]:.2e}t^2 + {coeffs[1]:.2e}t + {coeffs[2]:.2e}"

                            multiline_value = f"Method: {trend}<br>Equation: {equation}"
                            update_action(clock_name, 'Detrend', multiline_value)
                        

                        if clock_name in st.session_state.offset_selection:
                            offset_method = st.session_state.offset_selection[clock_name]
                            mean_before = st.session_state.offset_means_before.get(clock_name)
                            if mean_before is not None:
                                if offset_method == "Remove Offset[Mean Value]":
                                    mean_after = st.session_state.offset_means_after.get(clock_name)
                                    if mean_after is not None:
                                        offset_info = f"Method: {offset_method}<br>Mean Before [{st.session_state.order_of_data}]: {mean_before:.2e}<br>Mean After [{st.session_state.order_of_data}]: {mean_after:.2e}"
                                    else:
                                        offset_info = f"Method: {offset_method}<br>Mean Before [{st.session_state.order_of_data}]: {mean_before:.2e}"
                                else:
                                    offset_info = f"Method: {offset_method}<br>Mean Before [{st.session_state.order_of_data}]: {mean_before:.2e}"
                            else:
                                offset_info = f"Method: {offset_method}<br>Mean Before [{st.session_state.order_of_data}]: N/A"
                            update_action(clock_name, 'Offset Removed', offset_info)
                                                

                        if clock_name in st.session_state.outlier_selection:
                            outlier_method = st.session_state.outlier_selection[clock_name]
                            if outlier_method == "Std_Dev Based":
                                std_threshold = st.session_state.std_threshold[clock_name]
                                outlier_info = f"Method: {outlier_method}<br>Multiplier: {std_threshold:.2e}"
                            else:
                                outlier_info = f"Method: {outlier_method}"
                            update_action(clock_name, 'Outlier Filtered', outlier_info)
                            

                        if clock_name in st.session_state.smoothing_method:
                            smoothing_method = st.session_state.smoothing_method[clock_name]
                            window_size = st.session_state.window_size[clock_name]
                            smoothing_info = f"Method: {smoothing_method}<br>Window Size: {window_size}"
                            update_action(clock_name, 'Smoothed', smoothing_info)
                            

                        # Check if stability results exist for the clock under each analysis type
                        for analysis_type in st.session_state.stability_results:
                            if clock_name in st.session_state.stability_results[analysis_type]:
                                # st.write(f"{clock_name} stability results available for {analysis_type}:",
                                        # st.session_state.stability_results[analysis_type][clock_name])
                                results = st.session_state.stability_results[analysis_type][clock_name]
                                for result in results:
                                    tau, stability = result.split(',')
                                    # st.write(f"Appending result for {clock_name} - {analysis_type}: {tau}, {stability}")
                                    outcome_data.append([clock_name, analysis_type, int(tau), float(stability)])
                            # else:
                                # st.write(f"No stability results for {clock_name} under {analysis_type}")

                    transform_and_display(selected_clock_names)

            else:
                st.warning("Please go to the Raw Data tab and select the clocks you want to analyse and click the button Process Selected Clocks",icon="⚠️")


if __name__ == "__main__":
    run_once = 0
    main()
 
