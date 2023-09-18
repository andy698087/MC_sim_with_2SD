"""
start_time: 2023-09-03 17:12:31.515876
end_time: 2023-09-03 17:16:15.667721
time_difference: 0:03:44.151845

nMonte = 10000; N1 = 50; CV= 0.5

ln ratio SE include zero: 0.9174
Count (ratio) P < 0.05: 826 (0.082600)
Count (ratio) P < 0.01: 203 (0.020300)
Percentile 2.5 > 0 count (ratio): 416 (0.041600)
Percentile 97.5 > 0 count (ratio): 440 (0.044000)
"""

import pandas as pd
import os
import re

# Define a regular expression pattern to match content within parentheses
pattern = r'\((.*?)\)'

# Define the directory where your text files are located
directory = "C:/Users/User/MC_sim_2SD/GPM_MC_2SD_higher_orders_2_compare_20230916"

# Get a list of files that match the pattern in the directory
matching_files = [file for file in os.listdir(directory) if file.startswith("GPM_MC_nMonte_100000_N") and file.endswith(".txt")]

# Initialize an empty list to store data from all files

data = {'MethodOfMoments':[]}
# Loop through the matching files and extract data
for filename in matching_files:
    # Read data from the text file
    file_path = os.path.join(directory, filename)
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # if  == 'noMethodOfMoments.txt' or filename.split('_')[-1] == 'False.txt':
    #     data['MethodOfMoments'].append('No_MethodOfMoments')
    # elif filename.split('_')[-1] == 'Higgins1.txt':
    #     data['MethodOfMoments'].append('Higgins1')
    # elif filename.split('_')[-1] == 'Higgins2.txt':
    #     data['MethodOfMoments'].append('Higgins2')
    # else:
    #     data['MethodOfMoments'].append('Orignal_MethodOfMoments')
    # Extract relevant information from each file
    data['MethodOfMoments'].append('_'.join(filename.rstrip('.txt').split('_')[9:]))
    for n, line in enumerate(lines):
        if n == 4:
            key_values = line.strip().split('; ')
            for key_value in key_values:
                key, value = key_value.strip().split('= ')
                if key in data.keys():
                    data[key].append(value.strip())
                else:
                    data[key] = [value.strip()]
        if n > 4:
            if len(line.strip().split(':')) > 1:
                key, value = line.strip().split(':')
                if '(' in key:
                    key1 = re.sub(pattern, '', key)
                    key2_ = re.sub(pattern, '', key)
                    key2 = re.sub(r'count', '', key2_, flags=re.IGNORECASE) + ' ' + re.findall(pattern, key)[0]
                    value1 = re.sub(pattern, '', value)
                    # print(value1)
                    value2 = re.findall(pattern, value)[0]
                    # print(value2)
                    
                    if key1 in data.keys():
                        data[key1].append(value1)
                    else:
                        data[key1] = [value1]

                    if key2 in data.keys():
                        data[key2].append(value2)
                    else:
                        data[key2] = [value2]
                else:
                    if key in data.keys():
                        data[key].append(value.strip())
                    else:
                        data[key] = [value.strip()]

    # Append the extracted data to the list

# Create a DataFrame from the list of data
df = pd.DataFrame(data)
df[df.columns[1:]] = df[df.columns[1:]].astype(float)
# # Define the filename for your Excel file
excel_filename = "From_text_GPM_MC_nMonte_100000_N_higher_orders_2_compare_20230916.xlsx"

# # Create a Pandas Excel writer using XlsxWriter as the engine
# excel_writer = pd.ExcelWriter(excel_filename, engine="xlsxwriter")

# Convert the DataFrame to an XlsxWriter Excel object
df.to_excel(excel_filename, index=False)
quit()
# Get the xlsxwriter workbook and worksheet objects
workbook = excel_writer.book
worksheet = excel_writer.sheets["Sheet1"]

# Set the column widths to fit the content
for i, col in enumerate(df.columns):
    max_length = max(df[col].astype(str).map(len).max(), len(col))
    worksheet.set_column(i, i, max_length)

# # Close the Pandas Excel writer and save the Excel file
# workbook.save()

print(f"Data saved to {excel_filename}")