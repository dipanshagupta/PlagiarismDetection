import sys
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import os
import Levenshtein
import argparse
from numpy import median

# Function to clean the data i.e. fill empty fields, etc
def cleanData(dataframe):
    # Fill empty values with noanswer
    dataframe.fillna(value='noanswer',inplace=True)
    # Convert everything to string
    dataframe = dataframe.apply(lambda x: x.apply(lambda x: str(x)))
    # Convert everything to lower case
    dataframe = dataframe.apply(lambda x: x.apply(lambda x: str(x).lower()))

def levDistance(str1, str2):
    if str(str1) == "noanswer" or str(str2) == "noanswer": return 0
    else: return Levenshtein.ratio(str(str1), str(str2))

# Function to calculate metric i.e. detec plagiarism
def calculateMetrics(dataframe):
    # Clean the data obtained
    cleanData(dataframe)
    
    # Obtain data as matrix, for easier traversal
    data = dataframe.as_matrix()
    
    # This matrix will store similarity between ith and jth student
    median_similarity = []
    
    # For every row i (or student i)
    for i in range(len(data)):
        medians_for_ith = []
        
        # For every row j (or student j)
        for j in range(len(data)):
            
            # If same row, skip
            if i == j:
                medians_for_ith.append(0)
                continue
            else:
                
                list = []
                # Find similarity ratio in every column i.e. ratio for all questions
                for k in range(len(data[i])):
                    ratio = levDistance(data[i][k], data[j][k])
                    list.append(ratio)
                
                # Find median for this list of ratios, this will give us the median score between these 2 student
                medians_for_ith.append(median(list))

        median_similarity.append(medians_for_ith)

print median_similarity

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



