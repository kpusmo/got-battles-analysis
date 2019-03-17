from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.externals.six import StringIO
import pydotplus
from sklearn.metrics import classification_report, confusion_matrix
from prepare_data import x_train, x_test, y_train, y_test, feature_names, labels
from roc_plot import print_roc_plot


def run_decision_tree():
    clf = tree.DecisionTreeClassifier()
    clf.fit(x_train, y_train)

    # Predict
    y_pred = clf.predict(x_test)

    # How accurate was classifier on testing set
    output = accuracy_score(y_test, y_pred)
    print("Accuracy of decission tree: {:.2f}".format(output))
    print(classification_report(y_test, y_pred))
    cm = confusion_matrix(y_test, y_pred)
    print("\tDecision tree confusion matrix:\n", cm, "\n")
    print_roc_plot(y_pred, clf, 'Decision Tree')

    dot_data = StringIO()
    tree.export_graphviz(
        decision_tree=clf,
        out_file=dot_data,
        feature_names=feature_names,
        class_names=labels,
        filled=True,
        rounded=True,
        impurity=True
    )
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    graph.write_pdf('out/battles-decision-tree.pdf')
