import sys
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import os
import Levenshtein
import argparse
from numpy import median
from sklearn.metrics.pairwise import cosine_similarity

# Function to clean the data i.e. fill empty fields, etc
def cleanData(dataframe):
    # Fill empty values with noanswer
    dataframe.fillna(value='noanswer', inplace=True)
    # Convert everything to string
    dataframe = dataframe.apply(lambda x: x.apply(lambda x: str(x)))
    # Convert everything to lower case
    dataframe = dataframe.apply(lambda x: x.apply(lambda x: str(x).lower()))


def levDistance(str1, str2):
    if str(str1) == "noanswer" or str(str2) == "noanswer":
        return 0
    else:
        return Levenshtein.ratio(str(str1), str(str2))


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
                    ratio = 1 - levDistance(data[i][k], data[j][k])
                    list.append(ratio)

                # Find median for this list of ratios, this will give us the median score between these 2 student
                medians_for_ith.append(median(list))

        median_similarity.append(medians_for_ith)

    return median_similarity


def getValuesAboveThreshold(matrix, threshold):
    for i in range(len(matrix)):
        for j in range(i, len(matrix)):
            if matrix[i][j] > threshold:
                print "Student", (i + 1), "and Student", (j + 1), "Score =", matrix[i][j]


def wierdness(dataframe):
    # Clean the data obtained
    cleanData(dataframe)

    # Obtain data as matrix, for easier traversal
    data = dataframe

    value_counts = []

    for i in data.columns:
        inverse_value_counts = (1 - data[i].value_counts() / data[i].count())
        value_counts.append(pd.Series(MinMaxScaler().fit_transform(inverse_value_counts.values.reshape(-1, 1)).reshape(-1, ), index=inverse_value_counts.index))

    final_df = pd.DataFrame()
    for i in range(0, data.shape[1]):
        temp = data[data.columns[i]].apply(lambda x: value_counts[i][x])
        final_df[i] = temp

    ##print final_df

    ## WHAT DOES THIS DO?
    columns_to_include = df.apply(lambda x: x.apply(lambda x: len(x.split(" ")) < 10).sum() == df.shape[0])
    mcq_scored_df = final_df.loc[:,(df.apply(lambda x: x.apply(lambda x: len(x.split(" ")) < 10).sum() == df.shape[0])).values]
    mcq_scored_df.columns = df.columns[columns_to_include]


    import copy
    test = copy.copy(mcq_scored_df)

    def f(x):
        if x < 0.8:
            return 0
        else:
            return x

    test1 = test.apply(lambda x: x.apply(lambda y: f(y)))
    similarities = cosine_similarity(test1)

    test_sims = copy.copy(similarities)
    test_sims.sort()
    ##  test_sims_df = pd.DataFrame(test_sims)

    similarities_ls = []
    for i in range(similarities.shape[0]):
        for j in range(i + 1, similarities.shape[1]):
            similarities_ls.append((i, j, similarities[i][j]))

    similarities_df = pd.DataFrame(data=similarities_ls, columns=['Student_1', 'Student_2', 'Score'])

    ##print(similarities_df)
    return similarities_df




def getweirdAboveThreshold(similaritiesdf, threshold):
    #top_cases = similarities_df.sort_values(by='Score', ascending=False, inplace=False)
    #print top_cases.head(n=numCases).groupby("Student_1").head(numCases)

    print '\n\nStudents with similarities higher than:', threshold
    if (similarities_df['Score'] > threshold).sum() != 0:
        print similarities_df.loc[similarities_df['Score'] > threshold, :]
    else:
        print '\tNo student above similarity threshold:', threshold

# Main code

parser = argparse.ArgumentParser(description='Plaigarism detection tool')
parser.add_argument('-f', '--file', help='File location of answers', required=True)
parser.add_argument('-ty', '--type', help='Type of file', required=True)
parser.add_argument('-t','--top', type=int, default=20, help='Number of Top Cases to show')

args = vars(parser.parse_args())

# Get file location and the typ of file
fileLocation = args['file']
type = args['type']
numCases = int(args['top'])

# This matrix will store similarity between ith and jth student
median_similarity = []

# If file is excel, use ExcelFile facility to convert into dataframe
if type == "excel":
    xl = pd.ExcelFile(fileLocation)
    df = xl.parse(xl.sheet_names[0])
    median_similarity = calculateMetrics(df)
    wierdness(df)
else:
    # If file is csv, use read_csv to convert into dataframe
    if type == "csv":
        df = pd.read_csv(fileLocation, error_bad_lines=False)
        median_similarity = calculateMetrics(df)
        similarities_df= wierdness(df)

getValuesAboveThreshold(median_similarity, 0.7)
getweirdAboveThreshold(similarities_df, 0.7)



