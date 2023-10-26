import math
import numpy as np
import pandas as pd
import seaborn as sns
import pickle
from collections import Counter
from imblearn.over_sampling import SMOTE
import sklearn
from matplotlib import pyplot
from scipy.stats import shapiro
from sklearn import model_selection
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, accuracy_score, plot_confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, GridSearchCV


def upload_data():
    ######  Data understanding and cleaning  #####
    dfaux = pd.read_csv('Proj1_Dataset.csv', sep=',')  # open the file
    print("SHAPE: ", dfaux.shape)  # check the data shape
    print("INFO:")
    print(dfaux.info())  # see the information regarding the data on the Dataframe
    dfaux.drop_duplicates(inplace=True)  # see if it has duplicates and removes them
    print("\nSHAPE: ", dfaux.shape)
    print("NUMBER OF NULL VALUES: \n", dfaux.isnull().sum())  # shows how many null values each column has
    return dfaux


def create_subset(df):
    df_subset_aux = df.loc[:, 'S1Temp':'Persons'].copy()  # creates data subset without first two columns
    df_subset_aux.interpolate(inplace=True)  # Finds null values and substitutes them with average of next two
    print("DATA INFORMATION: \n", df_subset_aux.head(5))
    # print(df[(df["CO2"].isnull())].index)
    return df_subset_aux


def remove_outliers(df_subset):
    # helps to understand the data by giving information about frequency of numbers, max and min
    print("FREQUENCY OF DATA:\n", df_subset['Persons'].value_counts().head(20))  # from here we can take
    # that data is not well-balanced
    print("max: ", df_subset['PIR2'].max())
    print("min: ", df_subset['PIR2'].min())

    # Sees and takes out the outliers
    k = 6
    for i in range(9):
        
        df_subset[features[i]].plot(kind="box");
        plt.show()
        plt.close()
        
        standard_deviation = df_subset[features[i]].std()  # calculates standard deviation of each feature
        mean = df_subset[features[i]].mean()  # calculates mean of each feature
        Outlier_max = k * standard_deviation  # calculates max value that is considered an outlier of each feature
        # print(mean)
        # print(standard_deviation)
        indexLow = df_subset[(df_subset[features[i]] <= (mean - Outlier_max)) | (df_subset[features[i]] >=
                    (mean + Outlier_max))].index  # Trys to find outliers and stores their index
        # print("------ ", i)
        # print(indexLow)
        df_subset.loc[indexLow, features[i]] = np.nan  # sets outliers value to null
        # print("DATA INFORMATION: \n", df_subset.head(3760))

    df_subset.interpolate(inplace=True)  # Finds null values and substitutes them with average of next two


def scatter_plot_comparison(df_subset):
    sns.relplot(data=df_subset, x='PIR1', y='S1Light', hue='Persons', size='CO2')
    plt.show()
    plt.close()
    # print("DATA INFORMATION: \n", df_subset.head(3760))


def multiclass_problem(neurons, X_train, y_train, X_test, y_test):
    # Testa hyperparametros para o training set normal, o melhor foi (activation='relu', hidden_layer_sizes=(6, 4,
    # 4), learning_rate='adaptive', max_iter=2000, random_state=4, solver='sgd'), o 2 melhor é {'activation': 'relu',
    # 'hidden_layer_sizes': (6, 3), 'learning_rate': 'constant', 'max_iter': 2000, 'random_state': 4, 'solver': 'sgd'}

    # change hyperparameters below and run the code to test which ones give the best results
    parameters = {'hidden_layer_sizes': [(neurons+1,3,3), (neurons+1,3,2), (neurons+1,4,4),
                                         (neurons+1,4,2), (neurons+1,5,5), (neurons+1,4,3)],
                  'max_iter': [1000, 2000],
                  'random_state': [4],
                  'activation': ['logistic', 'relu', 'tanh'],
                  'solver': ['sgd'],
                  'learning_rate': ['constant', 'adaptive']
                  }

    # lbfgs doesn't converge
    clf = GridSearchCV(MLPClassifier(), parameters, n_jobs=-1, scoring='f1_micro')  # does cross validation with the
    #parameters given before to check the best ones

    clf.fit(X_train, y_train)  # does gridsearch on the training data to create the best model
    print('Score:',clf.score(X_train, y_train))
    print('Best Params:',clf.best_params_)  # best parameters achieved by the gridsearch

    # Testa hyperparametros para o training set ao usar o método de sampling SMOTE to balance the data

    parameters1 = {
        'hidden_layer_sizes': [(15, 9), (15,10), (15,11),(15,9,9)],
        'max_iter': [1000],
        'random_state': [4],
        'activation': ['logistic'], 'solver': ['sgd'],
        'learning_rate': ['constant', 'adaptive']}

    sm = SMOTE(random_state=42)  # creates the smote model
    
    X_res, y_res = sm.fit_resample(X_train, y_train)  # resamples the data to make it balanced
    
    print('Counter:',Counter(y_res))
    clf = GridSearchCV(MLPClassifier(), parameters1, n_jobs=-1, scoring='f1_micro')
    clf.fit(X_res, y_res)
    print('Score:',clf.score(X_res, y_res))
    print('Best Params:',clf.best_params_)

    
    # Tests the best model achieved with the SMOTE data
    print("SMOTE VALUES: \n")
    clf = MLPClassifier(activation='logistic', hidden_layer_sizes=(15, 9), learning_rate='adaptive', max_iter=1000,
                        random_state=4, solver='sgd')
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    scores_multi(clf, y_pred)

    # Tests the best model achieved with the unbalanced normal data
    print("NORMAL VALUES:")
    clf = MLPClassifier(activation='relu', hidden_layer_sizes=(6, 4, 4), learning_rate='adaptive', max_iter=2000,
                        random_state=4, solver='sgd')
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    scores_multi(clf, y_pred)

    # save NN model to disk
    filename = 'finalized_model.sav'
    pickle.dump(clf, open(filename, 'wb'))


def scores_multi(clf, y_pred):
    print(clf.score(X_train, y_train))
    print('Accuracy: {:.2f}'.format(accuracy_score(y_test, y_pred)))
    fig = plot_confusion_matrix(clf, X_test, y_test, display_labels=clf.classes_)
    fig.figure_.suptitle("Confusion Matrix for Multiclass Classification")
    plt.show()
    plt.close()
    print("MULTI-CLASS PROBLEM SCORES: ")
    print(classification_report(y_test, y_pred))


def binary_problem(neurons, X_train, y_train, X_test, y_test):
    # Testa hyperparametros para o training set normal, o melhor foi {'activation': 'relu', 'hidden_layer_sizes': (6,
    # 4), 'learning_rate': 'constant', 'max_iter': 1000, 'random_state': 4, 'solver': 'sgd'}

    # Testa hyperparametros para o training set
    parameters = {'hidden_layer_sizes': [(neurons-1,5,4), (neurons-1,5,2), (neurons-1,4, 4),
                                         (neurons-1,4,3), (neurons-1,4,2), (neurons,4,4)],
                  'max_iter': [1000, 2000],
                  'random_state': [4],
                  'activation': ['logistic', 'relu'], 'solver': ['sgd'],
                  'learning_rate': ['constant', 'adaptive']}

    # lbfgs doesn't converge; best results with just one hidden layer
    clf = GridSearchCV(MLPClassifier(), parameters, n_jobs=-1, scoring='f1_micro')

    clf.fit(X_train, y_train)
    print('Score:',clf.score(X_train, y_train))
    print('Best Params:',clf.best_params_)

    print("NORMAL VALUES:")
    clf = MLPClassifier(activation='relu', hidden_layer_sizes=(6, 4, 4), learning_rate='constant', max_iter=1000,
                        random_state=4, solver='sgd')
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    scores_bin(clf, y_pred)


def scores_bin(clf, y_pred):
    print(clf.score(X_train, y_train))
    print('Accuracy: {:.2f}'.format(accuracy_score(y_test, y_pred)))
    fig = plot_confusion_matrix(clf, X_test, y_test, display_labels=clf.classes_)
    fig.figure_.suptitle("Confusion Matrix for Binary Classification")
    plt.show()
    plt.close()
    print("BINARY PROBLEM SCORES: ")
    print(classification_report(y_test, y_pred))


######  Data understanding and cleaning  #####
df = upload_data()  # reads the data from the file

# Transforms Date and time data to correct form
df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
df['Time'] = pd.to_datetime(df['Time'])
df['Time'] = pd.to_datetime(df['Time'], format='%H:%M').dt.time

df_subset = create_subset(df)  # Creates the subset with specific features

# Select our features that we plan to use to predict the right Species
features = ['S1Temp', 'S2Temp', 'S3Temp', 'S1Light', 'S2Light', 'S3Light', 'CO2', 'PIR1', 'PIR2']

remove_outliers(df_subset)  # removes and substitutes the outliers from the subset

scatter_plot_comparison(df_subset)  # plots scatter plot of the variables

#####  From now on the data is taken care of, so it's time to make the NN algorithm  #####

# In this part we test the best hyperparameters, and we calculate the scores for the predictions
neurons = round((np.sqrt(4 * 9)))

### Multiclass problem
X = df_subset.loc[:, features].copy()  # creates the variable from the features that we use to predict the output
Y = df_subset.loc[:, 'Persons'].copy()  # is the feature that we plan to predict

X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=42, train_size=.7)  # splits the data into a
# train set that we use to create the model, and a test set that we use to check the final score of the model on
# unseen data

scaler = MinMaxScaler()  # used to scale the data to make it usable
X_train = scaler.fit_transform(X_train)  # fits the scaler to the X_train variable and scales it
X_test = scaler.transform(X_test)  # scales the X_test with the values fitted on X_train

#####  See plots to ckeck which data is usefull after it's scaled  #####
plt.figure(figsize=(7.5,35))
plt.subplot(911)
plt.scatter(X_train[:,0],y_train)
plt.title('Parameter: ' + features[0])
plt.subplot(912)
plt.scatter(X_train[:,1],y_train)
plt.title('Parameter: ' + features[1])
plt.subplot(913)
plt.scatter(X_train[:,2],y_train)
plt.title('Parameter: ' + features[2])
plt.subplot(914)
plt.scatter(X_train[:,3],y_train)
plt.title('Parameter: ' + features[3])
plt.subplot(915)
plt.scatter(X_train[:,4],y_train)
plt.title('Parameter: ' + features[4])
plt.subplot(916)
plt.scatter(X_train[:,5],y_train)
plt.title('Parameter: ' + features[5])
plt.subplot(917)
plt.scatter(X_train[:,6],y_train)
plt.title('Parameter: ' + features[6])
plt.subplot(918)
plt.scatter(X_train[:,7],y_train)
plt.title('Parameter: ' + features[7])
plt.subplot(919)
plt.scatter(X_train[:,8],y_train)
plt.title('Parameter: ' + features[8])
plt.show()
plt.close()

# save the scaler model to disk
filename2 = 'scaler.sav'
pickle.dump(scaler, open(filename2, 'wb'))

multiclass_problem(neurons, X_train, y_train, X_test, y_test)  # function responsible to predict how many people are inside the lab

### Binary problem
df_subset['Binary'] = df_subset['Persons'].apply(lambda x: 1 if x > 2 else 0)  # creates new column about how many
# people are inside the room. If it is 1 there are more than two people, if it's 0 there is less

X = df_subset.loc[:, features].copy()
Y = df_subset.loc[:, 'Binary'].copy()

X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=42, train_size=.7)

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

binary_problem(neurons, X_train, y_train, X_test, y_test)   # function responsible to predict if there is more or less than 2 people inside the room

