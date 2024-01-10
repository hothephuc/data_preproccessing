import argparse
import pandas as pd
import numpy as np
import math
from datetime import datetime

def extract_columns_with_missing_values(data):
    columns_with_missing = []
    # Iterate through the columns
    for col in data.columns:
        if data[col].isnull().any():  # Check if any missing values in the column
            columns_with_missing.append(col)
    # Extract the columns with missing values into a new data frame
    columns_with_missing_df = data[columns_with_missing]
    return columns_with_missing_df


# Function to count the number of rows with missing data
def count_rows_with_missing_data(data):
    rows_with_missing = 0
    #Iterate through the columns
    for row in range(len(data)):
        has_missing = False
        for value in data.iloc[row]:
            if pd.isna(value):
                has_missing = True
                break
        if has_missing:
            rows_with_missing +=1
    # Convert the list to a list of column names
    return rows_with_missing

#Fill in the missing value using mean, median (for numeric properties) and mode (for the categorical attribute).
def calculate_mean(data, column_name): # calculate mean of one numeric properties
    sum = 0
    count = 1
    for value in data[column_name]:
        if not pd.isna(value):
            sum+=value
            count+=1
    if count == 0:
        return None  # Avoid division by zero
    return sum/count


def calculate_median(data, column_name): # calculate median of one numeric properties
    list_of_value=[]
    for value in data[column_name]:
        if not pd.isna(value):
            list_of_value.append(value)
    list_of_value  = sorted(list_of_value)
    length = len(list_of_value)
    if length == 0:
        return 0
    if length%2 != 0:
        median =  list_of_value[((length+1)//2)-1]
    elif length%2 == 0:
        median = (list_of_value[length//2] + list_of_value[(length//2)-1])/2
    return median

def calculate_mode(data, column_name):
    list_of_values = []
    for value in data[column_name]:
        if not pd.isna(value):
            list_of_values.append(value)
    if not list_of_values:
        return None  # Handle the case when the list is empty
    # Count the occurrences of each unique value
    value_counts = {}
    for value in list_of_values:
        if value in value_counts:
            value_counts[value] += 1
        else:
            value_counts[value] = 1
    # Find the value with the highest count
    max_count = max(value_counts.values())
    mode_values = [value for value, count in value_counts.items() if count == max_count]
    return mode_values

def fill_missing_data_and_save(data, output_file, imputation_method='mean', column_names = 'all_attributes'):
    # Handle the case where imputation_method is not 'mean', 'median', or 'mode'
    if imputation_method not in ['mean', 'median', 'mode']:
        raise ValueError("Imputation method must be 'mean', 'median', or 'mode'.")

    # Iterate through numeric and categorical columns
    if column_names == 'all_attributes':
        for column_name in data.columns:
            if pd.api.types.is_numeric_dtype(data[column_name]):
                # Numeric property
                if imputation_method == 'mean':
                    data[column_name].fillna(calculate_mean(data, column_name), inplace=True)
                elif imputation_method == 'median':
                    data[column_name].fillna(calculate_median(data, column_name), inplace=True)
            else:
                # Categorical property
                if imputation_method == 'mode':
                    data[column_name].fillna(calculate_mode(data, column_name)[0], inplace=True)
    else:
        column_name = column_names
        if pd.api.types.is_numeric_dtype(data[column_name]):
            # Numeric property
            if imputation_method == 'mean':
                data[column_name].fillna(calculate_mean(data, column_name), inplace=True)
            elif imputation_method == 'median':
                data[column_name].fillna(calculate_median(data, column_name), inplace=True)
        else:
            # Categorical property
            if imputation_method == 'mode':
                data[column_name].fillna(calculate_mode(data, column_name)[0], inplace=True)

    # Generate a timestamp for the output filename
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    # Create a filename that includes the timestamp, imputation method, and original filename
    output_filename = f"{output_file.split('.')[0]}_{imputation_method}_{column_names}_{timestamp}.csv"

    # Save the DataFrame to a new CSV file
    data.to_csv(output_filename, index=False)

def delete_rows_with_missing_values(data, threshold=0.5):
    # Calculate the threshold number of missing values based on the percentage
    num_attributes = len(data.columns)
    threshold_count = int(threshold * num_attributes)
    
    # Filter rows based on the threshold
    rows_to_delete = []
    
    for index, row in data.iterrows():
        missing_count = 0
        for value in row:
            if pd.isna(value):
                missing_count += 1
        
        if missing_count >= threshold_count:
            rows_to_delete.append(index)
    
    data_filtered = data.drop(rows_to_delete)
    
    return data_filtered

def delete_columns_with_missing_values(data, threshold=0.5):
    # Calculate the threshold number of missing values based on the percentage
    num_samples = len(data)
    threshold_count = int(threshold * num_samples)

    # Create a list of columns to delete
    columns_to_delete = []

    for column in data.columns:
        missing_count = 0
        for value in data[column]:
            if pd.isna(value):
                missing_count += 1
        if missing_count > threshold_count:
            columns_to_delete.append(column)

    # Drop the columns based on the threshold
    data_filtered = data.drop(columns=columns_to_delete)

    return data_filtered



def delete_duplicate_samples(data):
    unique_rows = []

    for i in range(len(data)):
        is_duplicate = False
        for j in range(i+1, len(data)):
            # Compare columns in both rows
            is_duplicate = True  # Initialize is_duplicate to False
            for col in data.columns:
                if pd.isna(data.iloc[i][col]):
                    if not pd.isna(data.iloc[j][col]):
                        is_duplicate = False
                        break
                elif pd.isna(data.iloc[j][col] ):
                    if not pd.isna(data.iloc[i][col] ):
                        is_duplicate = False
                        break
                elif data.iloc[i][col] != data.iloc[j][col]:
                    is_duplicate = False
                    break

            if is_duplicate:
                break

        if not is_duplicate:
            unique_rows.append(data.iloc[i])

    return pd.DataFrame(unique_rows, columns=data.columns)

# 7. Normalize a numeric attribute using min-max and Z-score methods.
def min_max_scaling(data, column_name):
    # Extract the column for Min-Max scaling
    column = data[column_name]
    
    # Calculate Min-Max scaling
    min_val = column.min()
    max_val = column.max()
    if min_val == max_val:
        raise ValueError("cannot scale because min and max values are equal")
    for i in range(len(column)):
        if not pd.isna(column.iloc[i]):
            column.iloc[i] = (column.iloc[i] - min_val) / (max_val - min_val)
    data_scaled = data.assign(column_name=column)
    return data_scaled

def calculate_standard_deviation(data, column_name):
    mean = calculate_mean(data, column_name)
    if mean is None:
        return None  # Unable to calculate standard deviation without a mean

    sum_squared_diff = 0
    count = 0
    for value in data[column_name]:
        if not pd.isna(value):
            sum_squared_diff += (value - mean) ** 2
            count += 1
    if count < 2:
        return None  # Insufficient data for standard deviation
    return math.sqrt(sum_squared_diff / (count - 1))

def z_score_standardization(data, column_name):
    # Extract the column for Z-score standardization
    column = data[column_name]
    
    # Calculate Z-score standardization
    mean_val = calculate_mean(data, column_name)
    std_dev = calculate_standard_deviation(data, column_name)
    for i in range(len(column)):
        if not pd.isna(column.iloc[i]):
            column.iloc[i] = (column.iloc[i] - mean_val) / std_dev
    data_scaled = data.assign(column_name=column)
    
    return  data_scaled 

#8
def add_attributes(data, attribute1, attribute2):
    if (pd.api.types.is_numeric_dtype(data[attribute1]) and pd.api.types.is_numeric_dtype(data[attribute2])):
        data[attribute1].fillna(calculate_mean(data, attribute1), inplace=True)
        data[attribute2].fillna(calculate_mean(data, attribute2), inplace=True)
        return data[attribute1] + data[attribute2]
    else:
        return 'Error: Both attributes should be numeric (int64 or float64)'

def sub_attributes(data, attribute1, attribute2):
    if (pd.api.types.is_numeric_dtype(data[attribute1]) and pd.api.types.is_numeric_dtype(data[attribute2])):
        data[attribute1].fillna(calculate_mean(data, attribute1), inplace=True)
        data[attribute2].fillna(calculate_mean(data, attribute2), inplace=True)
        return data[attribute1] - data[attribute2]
    else:
        return 'Error: Both attributes should be numeric (int64 or float64)'

def mul_attributes(data, attribute1, attribute2):
    if (pd.api.types.is_numeric_dtype(data[attribute1]) and pd.api.types.is_numeric_dtype(data[attribute2])):
        data[attribute1].fillna(calculate_mean(data, attribute1), inplace=True)
        data[attribute2].fillna(calculate_mean(data, attribute2), inplace=True)
        return data[attribute1] * data[attribute2]
    else:
        return 'Error: Both attributes should be numeric (int64 or float64)'

def div_attributes(data, attribute1, attribute2):
    if (pd.api.types.is_numeric_dtype(data[attribute1]) and pd.api.types.is_numeric_dtype(data[attribute2])):
        data[attribute1].fillna(calculate_mean(data, attribute1), inplace=True)
        data[attribute2].fillna(calculate_mean(data, attribute2), inplace=True)
        return data[attribute1] + data[attribute2]
    else:
        return 'Error: Both attributes should be numeric (int64 or float64)'

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Data Preprocessing Tool")
    parser.add_argument("input_file", help="Input CSV file")
    parser.add_argument("function_select", help="Select function ")
    parser.add_argument("--method", help="Imputation method (mean, median, mode)")
    parser.add_argument("--column", help="Column to impute")
    parser.add_argument("--threshold", help="Threshold for deletion of rows or columns")
    parser.add_argument("--operation", help="addition, subtraction, multiplication, division of two columns")
    parser.add_argument("--attr1", help="first col")
    parser.add_argument("--attr2", help="second")

    args = parser.parse_args()

    data = pd.read_csv(args.input_file)
    
    if args.function_select == '1':
        print(extract_columns_with_missing_values(data))
        missing_values= extract_columns_with_missing_values(data)
        timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        # Create a filename that includes the timestamp, imputation method, and original filename
        output_filename = f"{args.input_file.split('.')[0]}_extract_columns_with_missing_values_{args.column}_{timestamp}.csv"
        missing_values.to_csv(output_filename, index=False)
    
    if args.function_select == '2':
        print("Number of instances with missing values:",count_rows_with_missing_data(data))

    if args.function_select == '3':
        if not args.column:
            fill_missing_data_and_save(data, args.input_file, args.method)
        else:
            fill_missing_data_and_save(data,args.input_file, args.method, column_names= args.column )
    
    if args.function_select == '4':
        if not args.threshold:
            filtered_dat = delete_rows_with_missing_values(data)
            timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
            # Create a filename that includes the timestamp, imputation method, and original filename
            output_filename = f"{args.input_file.split('.')[0]}_filtered_Rows_0.5_{timestamp}.csv"
            print(filtered_dat)
            filtered_dat.to_csv(output_filename, index=False)
        else:
            filtered_dat = delete_rows_with_missing_values(data, float(args.threshold))
            timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
            # Create a filename that includes the timestamp, imputation method, and original filename
            output_filename = f"{args.input_file.split('.')[0]}_filtered_Rows_{args.threshold}_{timestamp}.csv"
            print(filtered_dat)
            filtered_dat.to_csv(output_filename, index=False)

    if args.function_select == '5':
        if not args.threshold:
            filtered_dat = delete_columns_with_missing_values(data)
            timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
            # Create a filename that includes the timestamp, imputation method, and original filename
            output_filename = f"{args.input_file.split('.')[0]}_filtered_col_0.5_{timestamp}.csv"
            print(filtered_dat)
            filtered_dat.to_csv(output_filename, index=False)
        else:
            filtered_dat = delete_columns_with_missing_values(data, float(args.threshold))
            timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
            # Create a filename that includes the timestamp, imputation method, and original filename
            output_filename = f"{args.input_file.split('.')[0]}_filtered_col_{args.threshold}_{timestamp}.csv"
            print(filtered_dat)
            filtered_dat.to_csv(output_filename, index=False)

    if args.function_select == '6':
        filtered_dat = delete_duplicate_samples(data)
        timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        # Create a filename that includes the timestamp, imputation method, and original filename
        output_filename = f"{args.input_file.split('.')[0]}_filtered_duplicates_{timestamp}.csv"
        print(filtered_dat)
        filtered_dat.to_csv(output_filename, index=False)


    if args.function_select == '7':
        if args.method == 'min_max':
            scaled_data = min_max_scaling(data, args.column)
            timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
            # Create a filename that includes the timestamp, imputation method, and original filename
            output_filename = f"{args.input_file.split('.')[0]}_scale_min_max_{args.column}_{timestamp}.csv"
            print(scaled_data)
            scaled_data.to_csv(output_filename, index=False)
        else:
            scaled_data = z_score_standardization(data, args.column)
            timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
            # Create a filename that includes the timestamp, imputation method, and original filename
            output_filename = f"{args.input_file.split('.')[0]}_scale_z_score_{args.column}_{timestamp}.csv"
            print(scaled_data)
            scaled_data.to_csv(output_filename, index=False)
    if args.function_select == '8':
        if args.operation == 'add':
            result = add_attributes(data, args.attr1, args.attr2)
        elif args.operation == 'subtract':
            result = sub_attributes(data, args.attr1, args.attr2)
        elif args.operation == 'multiply':
            result = mul_attributes(data, args.attr1, args.attr2)
        elif args.operation == 'divide':
            result = div_attributes(data, args.attr1, args.attr2)
        else:
            result = 'Invalid operation'
    
        if isinstance(result, pd.Series):
            # Save the result to a new CSV file
            timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
            output_filename = f"{args.input_file.split('.')[0]}_{args.operation}_{args.attr1}_{args.attr2}_{timestamp}.csv"
            result.to_csv(output_filename, index=False)
        else:
            print(result)


#import pandas as pd
#data = pd.read_csv('house-prices.csv')
##print(data.isnull().any().to_list())
#data_filtered = data.drop_duplicates()
#
#timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
#        # Create a filename that includes the timestamp, imputation method, and original filename
#output_filename = f"{'house-prices.csv'.split('.')[0]}_filtered_duplicates_{timestamp}.csv"
#data_filtered.to_csv(output_filename, index=False)
