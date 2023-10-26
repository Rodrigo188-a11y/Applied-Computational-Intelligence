import numpy as np
import pandas as pd
import seaborn as sns
import skfuzzy as fuzz
import matplotlib.pyplot as plt
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, accuracy_score, plot_confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, GridSearchCV


def upload_data():
    ######  Data understanding and cleaning  #####
    dfaux = pd.read_csv('Proj1_Dataset.csv', sep=',')  # open the file
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
        indexLow = df_subset[(df_subset[features[i]] <= (mean - Outlier_max)) | (df_subset[features[i]] >=
                            (mean + Outlier_max))].index  # Trys to find outliers and stores their index
        df_subset.loc[indexLow, features[i]] = np.nan  # sets outliers value to null
    df_subset.interpolate(inplace=True)  # Finds null values and substitutes them with average of next two

    # df_subset['Date'] = df.loc[:, 'Date']  # creates data subset without first two columns
    # df_subset['Time'] = df.loc[:, 'Time']  # creates data subset without first two columns
    # df_subset['Time'] = df_subset['Time'].astype(str)  # creates data subset without first two columns

    # creates data subset with all temperature values to diminute the amount of features
    df_subset['Temp'] = df_subset.loc[:, 'S1Temp'] + df_subset.loc[:, 'S2Temp'] + df_subset.loc[:, 'S3Temp']
    # creates data subset with all light values to diminute the amount of features
    df_subset['Light'] = df_subset.loc[:, 'S1Light'] + df_subset.loc[:, 'S2Light'] + df_subset.loc[:, 'S3Light']
    # creates data subset with all PIR values to diminute the amount of features
    df_subset['PIR'] = df_subset.loc[:, 'PIR1'] + df_subset.loc[:, 'PIR2']

    remove_outliers2()


def remove_outliers2():
    features2 = ['Temp', 'Light']
    k = 6
    for i in range(2):
        standard_deviation = df_subset[features[i]].std()  # calculates standard deviation of each feature
        mean = df_subset[features[i]].mean()  # calculates mean of each feature
        Outlier_max = k * standard_deviation  # calculates max value that is considered an outlier of each feature
        indexLow = df_subset[(df_subset[features[i]] <= (mean - Outlier_max)) | (df_subset[features[i]] >=
                                                                                 (
                                                                                             mean + Outlier_max))].index  # Trys to find outliers and stores their index
        df_subset.loc[indexLow, features[i]] = np.nan  # sets outliers value to null
    df_subset.bfill(axis='rows')


def scatter_plot_comparison():
    # df_subset.plot.bar(x='Binary', y='S1Temp', rot=0)
    sns.relplot(data=df_subset, x='Temp', y='Light', hue='Binary')
    # sns.relplot(data=df_subset, x='Temp', y='PIR', hue='Binary')
    # sns.relplot(data=df_subset, x='Light', y='PIR', hue='Binary')
    # plt.hist(df_subset['CO2'])
    # print("DATA INFORMATION: \n", df_subset.head(10))
    # print(type(df_subset['Time']))
    # df_subset['CO2'].plot()
    # that data is not well-balanced
    print("max: ", df_subset['Light'].max())
    print("min: ", df_subset['Light'].min())
    plt.show()
    plt.close()
    # print("DATA INFORMATION: \n", df_subset.head(3760))


def fuzzy_plot():
    temperature.view()
    plt.show()
    plt.close()
    light.view()
    plt.show()
    plt.close()
    pir.view()
    plt.show()
    plt.close()
    persons.view()
    plt.show()
    plt.close()


def fuzzy_defuzzification():
    binary_pred = []  # stores predicted results
    for i in range(len(binary_data)):
        # Retrieves data from the dataframe and estimates in the fuzzy rules
        person_estimate.input['temperature'] = temperature_data[i]
        person_estimate.input['light'] = light_data[i]
        person_estimate.input['pir'] = pir_data[i]

        person_estimate.compute()  # calculates the output

        # If the result is smaller than 1, there are 2 or fewer people on the room, if it's bigger there are more
        if person_estimate.output['persons'] <= 1:
            binary_pred.append(0)
        else:
            binary_pred.append(1)

    # Calculates value to do the classification report
    TP0, FP0, TN0, FN0, TP1, FP1, TN1, FN1 = 0, 0, 0, 0, 0, 0, 0, 0
    for i in range(len(binary_data)):
        if binary_data[i] == 0:
            if binary_pred[i] == 0:
                TP0 += 1
                TN1 += 1
            else:
                FP0 += 1
                FN1 += 1
        else:
            if binary_pred[i] == 1:
                TN0 += 1
                TP1 += 1
            else:
                FN0 += 1
                FP1 += 1

    precision0 = TP0 / (TP0 + FP0)
    recall0 = TP0 / (TP0 + FN0)
    f1score0 = 2 * ((precision0 * recall0) / (precision0 + recall0))

    acc = (TP0 + TN0) / len(binary_data)
    precision1 = TP1 / (TP1 + FP1)
    recall1 = TP1 / (TP1 + FN1)
    f1score1 = 2 * ((precision1 * recall1) / (precision1 + recall1))
    macroavg_precision = (precision1 + precision0)/2
    macroavg_recall = (recall1 + recall0) / 2
    macroavg_f1score = (f1score1 + f1score0) / 2

    print("BINARY PROBLEM SCORES FUZZY: ")
    print("            precision   recall  f1-score \n")
    print("       0        %.2f     %.2f      %.2f" % (precision0, recall0, f1score0))
    print("       1        %.2f     %.2f      %.2f" % (precision1, recall1, f1score1))
    print("\naccuracy                         %.2f" % acc)
    print("macro avg       %.2f     %.2f      %.2f" % (macroavg_precision, macroavg_recall, macroavg_f1score))


def binary_problem():
    # Testa hyperparametros para o training set normal, o melhor foi {'activation': 'relu', 'hidden_layer_sizes': (5,
    # 4), 'learning_rate': 'adaptive', 'max_iter': 1000, 'random_state': 4, 'solver': 'sgd'}

    """
    parameters = {'hidden_layer_sizes': [(7, 5, 4),(7,4,4), (7,4,3), (5,4,2),
                                         (7, 4,2), (7,6, 4), (7,5,3)],
                  'max_iter': [1000, 2000],
                  'random_state': [4],
                  'activation': ['logistic', 'relu'], 'solver': ['sgd'],
                  'learning_rate': ['constant', 'adaptive']}

    # lbfgs doesn't converge; best results with just one hidden layer
    clf = GridSearchCV(MLPClassifier(), parameters, n_jobs=-1, scoring='f1_micro')

    clf.fit(X_train, y_train)
    print(clf.score(X_train, y_train))
    print(clf.best_params_)
    """

    clf = MLPClassifier(activation='relu', hidden_layer_sizes=(5, 4, 2), learning_rate='constant', max_iter=1000,
                        random_state=4, solver='sgd')
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    scores_bin(clf, y_pred)


def scores_bin(clf, y_pred):
    fig = plot_confusion_matrix(clf, X_test, y_test, display_labels=clf.classes_)
    fig.figure_.suptitle("Confusion Matrix for Binary Classification")
    plt.show()
    plt.close()
    print("BINARY PROBLEM SCORES NN: ")
    print(classification_report(y_test, y_pred))


######  Data understanding and cleaning  #####
df = upload_data()  # reads the data from the file

# Transforms Date and time data to correct form
df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
df['Time'] = pd.to_datetime(df['Time'])
df['Time'] = pd.to_datetime(df['Time'], format='%H:%M').dt.time

df_subset = create_subset()  # Creates the subset with specific features

# Select our features that we plan to use to predict the right Species
features = ['S1Temp', 'S2Temp', 'S3Temp', 'S1Light', 'S2Light', 'S3Light', 'CO2', 'PIR1', 'PIR2']

remove_outliers()  # removes and substitutes the outliers from the subset

df_subset['Binary'] = df_subset['Persons'].apply(lambda x: 1 if x > 2 else 0)  # creates new column about how many
# people are inside the room. If it is 1 there are more than two people, if it's 0 there is less

# scatter_plot_comparison()  # plots scatter plot of the variables

#####  FUZZY LOGIC PART  #####
# Generate universe variables based on retrieved information from plots

# Inputs
temperature = ctrl.Antecedent(np.arange(57, 67, 0.5), 'temperature')
light = ctrl.Antecedent(np.arange(0, 1451, 1), 'light')
pir = ctrl.Antecedent(np.arange(0, 3, 0.5), 'pir')

# Output
persons = ctrl.Consequent(np.arange(0, 2.5, 0.5), 'persons')

# Generate fuzzy membership functions
temperature['cold'] = fuzz.trapmf(temperature.universe, [57, 57, 60, 62])
temperature['warm'] = fuzz.trimf(temperature.universe, [60, 62.5, 65])
temperature['hot'] = fuzz.trapmf(temperature.universe, [63, 65, 67, 67])

light['low'] = fuzz.trapmf(light.universe, [0, 0, 200, 400])
light['normal'] = fuzz.trapmf(light.universe, [200, 400, 1000, 1200])
light['bright'] = fuzz.trapmf(light.universe, [1000, 1200, 1450, 1450])

pir['slow'] = fuzz.trimf(pir.universe, [0, 0, 0.5])
pir['medium'] = fuzz.trimf(pir.universe, [0.5, 1, 1.5])
pir['fast'] = fuzz.trimf(pir.universe, [1.5, 2, 2.5])

persons['less'] = fuzz.trimf(persons.universe, [0, 0, 1.5])
persons['more'] = fuzz.trimf(persons.universe, [0.5, 2, 2])

fuzzy_plot()  # plots the membership functions

# Create rules
# Temp VS Light
rule1 = ctrl.Rule(temperature['cold'] & (light['low'] | light['normal']), persons['less'])
rule2 = ctrl.Rule(temperature['warm'] & (light['low'] | light['bright']), persons['more'])
rule3 = ctrl.Rule(temperature['warm'] & light['normal'], persons['less'])
rule4 = ctrl.Rule(temperature['hot'] & (light['low'] | light['normal'] | light['bright']), persons['more'])
# rule5 = ctrl.Rule(temperature['hot'] & light['bright'], persons['more'])

# Temp VS PIR
rule5 = ctrl.Rule(temperature['cold'] & (pir['slow'] | pir['medium'] | pir['fast']), persons['less'])
rule6 = ctrl.Rule(temperature['warm'] & pir['slow'], persons['less'])
rule7 = ctrl.Rule((temperature['hot']) & (pir['slow'] | pir['medium'] | pir['fast']), persons['more'])
rule8 = ctrl.Rule((temperature['warm']) & (pir['medium'] | pir['fast']), persons['more'])
plt.show()

# Light VS PIR
rule9 = ctrl.Rule(light['normal'] & (pir['slow'] | pir['medium'] | pir['fast']), persons['less'])
rule10 = ctrl.Rule(light['low'] & (pir['medium'] | pir['fast']), persons['more'])
rule11 = ctrl.Rule(light['bright'] & (pir['slow'] | pir['medium'] | pir['fast']), persons['more'])

persons_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, rule9, rule10, rule11])
person_estimate = ctrl.ControlSystemSimulation(persons_ctrl)

### Multiclass problem
# Retrieves data from the dataframe that will be used to predict if the fuzzy rules work
temperature_data = df_subset.loc[:, 'Temp'].to_numpy()
light_data = df_subset.loc[:, 'Light'].to_numpy()
pir_data = df_subset.loc[:, 'PIR'].to_numpy()
binary_data = df_subset.loc[:, 'Binary'].to_numpy()

fuzzy_defuzzification()  # function that runs the fuzzy code and scores it

# NN classifier to compare values
features = ['Temp', 'Light', 'PIR']

X = df_subset.loc[:, features].copy()
Y = df_subset.loc[:, 'Binary'].copy()

X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=42, train_size=.7)
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

binary_problem()


