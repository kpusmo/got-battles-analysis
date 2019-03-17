from sklearn.linear_model import LogisticRegression
from prepare_data import x_train, x_test, y_train, y_test
from sklearn.metrics import classification_report, confusion_matrix

from roc_plot import print_roc_plot


def run_logistic_regression():
    logreg = LogisticRegression(solver='lbfgs')
    logreg.fit(x_train, y_train)
    y_pred = logreg.predict(x_test)
    print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(x_test, y_test)))
    print(classification_report(y_test, y_pred))
    cm = confusion_matrix(y_test, y_pred)
    print("\tLogistic regression confusion matrix:\n", cm)
    print_roc_plot(y_pred, logreg, 'Logistic Regression')
