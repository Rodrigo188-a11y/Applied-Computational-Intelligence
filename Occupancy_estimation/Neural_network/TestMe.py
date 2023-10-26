import pickle
import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, accuracy_score, plot_confusion_matrix


def upload_data():
    ######  Data understanding and cleaning  #####
    name = sys.argv[1]  # read the name of the file to read from the terminal
    dfaux = pd.read_csv(name, sep=',')  # open the file
    dfaux.drop_duplicates(inplace=True)  # see if it has duplicates and removes them
    return dfaux


def create_subset():
    df_subset_aux = df.loc[:, 'S1Temp':'Persons'].copy()  # creates data subset without first two columns
    df_subset_aux.interpolate(inplace=True)  # Finds null values and substitutes them with average of next two
    return df_subset_aux


def remove_outliers():
    # Sees and takes out the outliers
    k = 6
    for i in range(9):
        standard_deviation = df_subset[features[i]].std()  # calculates standard deviation of each feature
        mean = df_subset[features[i]].mean()  # calculates mean of each feature
        Outlier_max = k * standard_deviation  # calculates max value that is considered an outlier of each feature
        indexLow = df_subset[(df_subset[features[i]] <= (mean - Outlier_max)) | (df_subset[features[i]] >= (mean +
                            Outlier_max))].index  # Trys to find outliers and stores their index
        df_subset.loc[indexLow, features[i]] = np.nan  # sets outliers value to null
    df_subset.interpolate(inplace=True)  # Finds null values and substitutes them with average of next two


def multiclass_problem():
    y_pred = clf.predict(X)
    scores(clf, y_pred)


def scores(clf, y_pred):
    fig = plot_confusion_matrix(clf, X, Y, display_labels=clf.classes_)
    fig.figure_.suptitle("Confusion Matrix for Multiclass Classification")
    plt.show()
    plt.close()
    print("MULTI-CLASS PROBLEM SCORES: ")
    print(classification_report(Y, y_pred))


######  Data understanding and cleaning  #####
df = upload_data()  # reads the data from the file

# load the model from disk
filename = 'finalized_model.sav'
filename2 = 'scaler.sav'
clf = pickle.load(open(filename, 'rb'))  # NN model created on the Proj1.1.py trained on the data
scaler = pickle.load(open(filename2, 'rb'))  # Scaler model created on the Proj1.1.py trained on the data

# Transforms Date and time data to correct form
df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
df['Time'] = pd.to_datetime(df['Time'])
df['Time'] = pd.to_datetime(df['Time'], format='%H:%M').dt.time

df_subset = create_subset()  # Creates the subset with specific features

# Select our features that we plan to use to predict the right Species
features = ['S1Temp', 'S2Temp', 'S3Temp', 'S1Light', 'S2Light', 'S3Light', 'CO2', 'PIR1', 'PIR2']

remove_outliers()  # removes and substitutes the outliers from the subset

### Multiclass problem
X = df_subset.loc[:, features].copy()  # creates the variable from the features that we use to predict the output
Y = df_subset.loc[:, 'Persons'].copy()  # is the feature that we plan to predict

X = scaler.transform(X)  # transforms the data with the X_train values trained on Proj1.1.py

multiclass_problem()  # function responsible to predict how many people are inside the lab
