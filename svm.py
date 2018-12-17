# Starting code for UVA CS 4501 ML- SVM
__author__ = 'tcs9pk'

import numpy as np
from sklearn import svm, datasets, preprocessing
import pandas as pd
import random


# Attention: You're not allowed to use the model_selection module in sklearn.
#            You're expected to implement it with your own code.
# from sklearn.model_selection import GridSearchCV

class SvmIncomeClassifier:
    def __init__(self):
        random.seed(0)

    def load_data(self, csv_fpath):
        col_names_x = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status',
                     'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss',
                     'hours-per-week', 'native-country', 'label']
        col_names_y = ['label']

        numerical_cols = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss',
                          'hours-per-week']
        categorical_cols = ['workclass', 'education', 'marital-status', 'occupation', 'relationship',
                            'race', 'sex', 'native-country']

        # mode calcs for categoricals

        data = pd.read_csv(csv_fpath)

        data.columns = col_names_x
        #print(y.head())

        #del data['label']
        workClassMode = data['workclass'].mode()[0]
        #print(workClassMode)
        educationMode = data['education'].mode()[0]
        maritalStatusMode = data['marital-status'].mode()[0]
        occupationMode = data['occupation'].mode()[0]
        relationshipMode = data['relationship'].mode()[0]
        raceMode = data['race'].mode()[0]
        sexMode = data['sex'].mode()[0]
        nativeCountryMode = data['native-country'].mode()[0]

        data['workclass'] = data['workclass'].str.replace(' \\?', 'Private', regex=True)
        data['education'] = data['education'].str.replace('\\?', educationMode, regex=True)
        data['marital-status'] = data['marital-status'].str.replace('\\?', maritalStatusMode, regex=True)
        data['occupation'] = data['occupation'].str.replace('\\?', occupationMode, regex=True)
        data['relationship'] = data['relationship'].str.replace('?', relationshipMode, regex=True)
        data['race'] = data['race'].str.replace('\\?', raceMode, regex=True)
        data['sex'] = data['sex'].str.replace('\\?', sexMode, regex=True)
        data['native-country'] = data['native-country'].str.replace('\\?', nativeCountryMode, regex=True)

        le = preprocessing.LabelEncoder()
        for column_name in data.columns:
            if data[column_name].dtype == object:
                data[column_name] = le.fit_transform(data[column_name])
            else:
                pass

        for column_name in data.columns:
            data[column_name] = (data[column_name] - data[column_name].min()) / (data[column_name].max() - data[column_name].min())

        y = data['label']
        del data['label']

        # 1. Data pre-processing.
        # Hint: Feel free to use some existing libraries for easier data pre-processing.

        # for all the numerical columns we're going to replace missing w average then normalize
        # for all the categorical columns we're going to find the mode and replace missing vals w that
        return data, y
        pass
    def train_and_select_model(self, training_csv):
        x_train, y_labels = self.load_data(training_csv)
        #self.load_data(training_csv)
        #print(x_train.head(10))
        #print(y_labels.head(10))


        # 2. Select the best model with cross validation.
        # Attention: Write your own hyper-parameter candidates.
        param_set = [
                     {'kernel': 'linear', 'C': 1.0},
                     {'kernel': 'rbf'},
                     {'kernel': 'poly', 'degree': 3.0},
                     {'kernel': 'linear', 'C': 10.0},
        ]
        #pass

        full = pd.DataFrame(pd.concat([x_train, y_labels], axis=1, sort=False))

        frames = [full, full]
        double = pd.DataFrame(pd.concat(frames))

        x_train_array = x_train.values
        y_train_array = np.array(y_labels.values)


        doubleX = np.concatenate((x_train_array, x_train_array), axis=0)
        doubleY = np.concatenate((y_train_array, y_train_array), axis =0)

        doubleX = doubleX.astype(float)
        doubleY = doubleY.astype(float)

        #CV time...
        # time to double it up

        training_1_x = doubleX[0:25894,]
        training_1_y = doubleY[0:25894,]
        testing_1_x = doubleX[25894:38840,]
        testing_1_y = doubleY[25894:38840,]

        training_2_x = doubleX[12946:38840,]
        training_2_y = doubleY[12946:38840,]
        testing_2_x = doubleX[0:12946,]
        testing_2_y = doubleY[0:12946,]

        training_3_x = doubleX[25893:51787,]
        training_3_y = doubleY[25893:51787,]
        testing_3_x = doubleX[25893:51787,]
        testing_3_y = doubleY[25893:51787,]


        scores = []

        # LINEAR 1 ---> SOFTIE.
        linear_model1 = svm.SVC(kernel='linear', C=1.0).fit(training_1_x, training_1_y)

        scores.append(linear_model1.score(testing_1_x, testing_1_y))

        linear_model2 = svm.SVC(kernel='linear', C=1.0).fit(training_2_x, training_2_y)

        scores.append(linear_model2.score(testing_2_x, testing_2_y))
        linear_model3 = svm.SVC(kernel='linear', C=1.0).fit(training_3_x, training_3_y)

        scores.append(linear_model3.score(testing_3_x, testing_3_y))

        ## HARD LINEAR MODEL

        linear_model1_hard = svm.SVC(kernel='linear', C=10.0).fit(training_1_x, training_1_y)

        scores.append(linear_model1_hard.score(testing_1_x, testing_1_y))

        linear_model2_hard = svm.SVC(kernel='linear', C=10.0).fit(training_2_x, training_2_y)

        scores.append(linear_model2_hard.score(testing_2_x, testing_2_y))
        linear_model3_hard = svm.SVC(kernel='linear', C=10.0).fit(training_3_x, training_3_y)

        scores.append(linear_model3_hard.score(testing_3_x, testing_3_y))

        ## RBF BOI

        rbf_model1 = svm.SVC(kernel='rbf').fit(training_1_x, training_1_y)

        scores.append(rbf_model1.score(testing_1_x, testing_1_y))

        rbf_model2 = svm.SVC(kernel='rbf').fit(training_2_x, training_2_y)

        scores.append(rbf_model2.score(testing_2_x, testing_2_y))

        rbf_model3 = svm.SVC(kernel='rbf').fit(training_3_x, training_3_y)

        scores.append(rbf_model3.score(testing_3_x, testing_3_y))

        poly_model1 = svm.SVC(kernel='poly', C=1.0).fit(training_1_x, training_1_y)

        scores.append(poly_model1.score(testing_1_x, testing_1_y))

        poly_model2 = svm.SVC(kernel='poly', C=1.0).fit(training_2_x, training_2_y)

        scores.append(poly_model2.score(testing_2_x, testing_2_y))

        poly_model3 = svm.SVC(kernel='poly', C=1.0).fit(training_3_x, training_3_y)

        scores.append(poly_model3.score(testing_3_x, testing_3_y))

        for score in scores:
            print score

        #print(max(scores))
        #print(scores.index(max(scores)))

        # best model is rbf 3 !
        return rbf_model3, scores[8]
        #return best_model, best_score

    def predict(self, test_csv, trained_model):
        x_test, y_labels = self.load_data(test_csv)

        predictions = trained_model.predict(x_test)

        return predictions

    def output_results(self, predictions):
        # 3. Upload your Python code, the predictions.txt as well as a report to Collab.
        # Hint: Don't archive the files or change the file names for the automated grading.
        with open('predictions.txt', 'w') as f:
            for pred in predictions:
                if pred == 0:
                    f.write('<=50K\n')
                else:
                    f.write('>50K\n')


if __name__ == '__main__':
    training_csv = "salary.labeled.csv"
    testing_csv = "salary.2Predict.csv"
    clf = SvmIncomeClassifier()
    #load_data(clf, training_csv)

    trained_model, cv_score = clf.train_and_select_model(training_csv)
    clf.train_and_select_model(training_csv)

    print "The best model was scored %.2f" % cv_score
    predictions = clf.predict(testing_csv, trained_model)
    clf.output_results(predictions)


