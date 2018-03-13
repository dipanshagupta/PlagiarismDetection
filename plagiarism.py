import sys
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import os
import argparse

# Function to clean the data i.e. fill empty fields, etc
def cleanData(dataframe):
    # Fill empty values with noanswer
    dataframe.fillna(value='noanswer',inplace=True)
    # Convert everything to string
    dataframe = dataframe.apply(lambda x: x.apply(lambda x: str(x)))
    # Convert everything to lower case
    dataframe = dataframe.apply(lambda x: x.apply(lambda x: str(x).lower()))

# Function to calculate metric i.e. detec plagiarism
def calculateMetrics(dataframe):
    # Clean the data obtained
    cleanData(dataframe)
    for i in dataframe.columns:
        value_counts = dataframe[i].value_counts()
        print value_counts


# Main code

parser = argparse.ArgumentParser(description='Plaigarism detection tool')
parser.add_argument('-f','--file', help='File location of answers', required=True)
parser.add_argument('-ty','--type', help='Type of file', required=True)

args = vars(parser.parse_args())

# Get file location and the typ of file
fileLocation = args['file']
type = args['type']

# If file is excel, use ExcelFile facility to convert into dataframe
if type == "excel":
    xl = pd.ExcelFile(fileLocation)
    df = xl.parse(xl.sheet_names[0])
    calculateMetrics(df)
else:
    # If file is csv, use read_csv to convert into dataframe
    if type == "csv":
        df = pd.read_csv(fileLocation)
        calculateMetrics(df)



