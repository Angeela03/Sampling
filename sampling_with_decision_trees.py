import pandas as pd
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV


def grid_search(train_x, train_y):
    # Select the best parameters for Decision tree
    clf = DecisionTreeClassifier()
    param_grid = {"min_samples_split": [2, 3, 5, 7],
                  "max_depth": [3, 5, 8, 10, 15, None],
                  "max_leaf_nodes": [3, 5, 10, 15, 20],
                  "min_samples_leaf": [1, 2, 3, 5]
                  }
    CV_rfc = GridSearchCV(estimator=clf, param_grid=param_grid, cv=5)
    CV_rfc.fit(train_x, train_y)
    print(CV_rfc.best_params_)


def create_x_y():
    # read_file reads the file from a csv file and splits into training and tests
    label = read_file.loc[:, 37].values
    features = read_file.loc[:, :36].values
    return features, label


def manipulate_data(clf, x_test, y_test):

    prediction = clf.predict(x_test)
    print("Accuracy:", accuracy_score(y_test, prediction))
    print("F1 score:", f1_score(y_test, prediction))


def main():
    # Creates data_x and data_y
    data_x, data_y = create_x_y()

    # Split the data into training and test sets
    x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size=0.25, random_state=9)

    # Performs grid search to select the best parameters for the decision tree model
    print("The best parameters for the model are:")
    grid_search(x_train, y_train)

    print("Fitting the model with original data:")
    clf = DecisionTreeClassifier(max_depth=5, max_leaf_nodes=10, min_samples_leaf=1, min_samples_split=2, random_state=44)
    clf.fit(x_train, y_train)
    manipulate_data(clf, x_test, y_test)

    print("Fitting the model after under sampling:")
    sm1 = RandomUnderSampler(random_state=7, ratio=0.5)
    x_train1, y_train1 = sm1.fit_sample(x_train, y_train)
    clf.fit(x_train1, y_train1)
    manipulate_data(clf, x_test, y_test)

    print("Fitting the model after over sampling:")
    sm2 = RandomOverSampler(random_state=7, ratio=0.5)
    x_train2, y_train2 =sm2.fit_sample(x_train, y_train)
    clf.fit(x_train2, y_train2)
    manipulate_data(clf, x_test, y_test)

    print("Fitting the model after performing SMOTE:")
    sm3 = SMOTE(random_state=7, ratio=0.5)
    x_train3, y_train3 = sm3.fit_sample(x_train, y_train)
    clf.fit(x_train3, y_train3)
    manipulate_data(clf, x_test, y_test)


if __name__ == '__main__':
    main()

