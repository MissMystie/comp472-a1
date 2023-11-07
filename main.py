import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.metrics import confusion_matrix, classification_report, f1_score, accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split, GridSearchCV


def load_dataset_penguins(dataset_filepath: str) -> pd.DataFrame:
    # Load dataset
    df = pd.read_csv(dataset_filepath)

    # Convert string features into 1-hot vectors
    df = pd.get_dummies(df, columns=['island'], dtype=int)
    df = pd.get_dummies(df, columns=['sex'], drop_first=True, dtype=int)

    return df


def load_dataset_abalone(dataset_filepath: str) -> pd.DataFrame:
    df = pd.read_csv(dataset_filepath)
    return df


def plot(df: pd.DataFrame, target: str, label_x: str, output_filepath: str):

    # Calculate the percentage of instances for output class ('species')
    class_percent = df[target].value_counts(normalize=True) * 100

    # Create a bar plot
    plt.figure(figsize=(8, 6))
    class_percent.plot(kind='bar', color='skyblue')
    plt.title('Percentage of Instances in Each Output Class')
    plt.xlabel(label_x)
    plt.ylabel('Percentage')
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Save the plot in a PDF file
    with PdfPages(output_filepath) as pdf:
        pdf.savefig()

    # Show the plot
    plt.show()


def base_dt(df: pd.DataFrame, X_train, X_test, y_train, y_test,  X):

    # Train DT classifier with default parameters
    base_dt = DecisionTreeClassifier()
    base_dt.fit(X_train, y_train)

    base_dt_accuracy = base_dt.score(X_test, y_test)
    print(f"Accuracy of Base-DT: {base_dt_accuracy:.2f}")

    y_pred = base_dt.predict(X_test)

    # Visualize
    plt.figure(figsize=(15, 10))
    plot_tree(base_dt, feature_names=X.columns, class_names=base_dt.classes_, filled=True)
    plt.show()

    return y_pred


def top_dt(df: pd.DataFrame, X_train, X_test, y_train, y_test):

    # Define hyperparameter grid
    param_grid = {
        'criterion': ['gini', 'entropy'],
        'max_depth': [None, 5, 10],
        'min_samples_split': [2,5,10]
    }

    base_dt = DecisionTreeClassifier()

    # Grid search and get best-performing DT
    grid_search = GridSearchCV(base_dt, param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)

    # Evaluate Top-DT
    top_dt = grid_search.best_estimator_
    top_dt_accuracy = top_dt.score(X_test, y_test)
    print(f"Accuracy of Top-DT: {top_dt_accuracy:.2f}")

    y_pred = top_dt.predict(X_test)

    return y_pred


def base_mlp(df: pd.DataFrame, X_train, X_test, y_train, y_test):

    base_mlp = MLPClassifier(
        hidden_layer_sizes=(100, 100),  # 2 hidden layers with 100 neurons each
        activation='logistic',  # Sigmoid (logistic) activation function
        solver='sgd',  # Stochastic gradient descent
        random_state=36  # For reproducibility
    )

    base_mlp.fit(X_train, y_train)

    # Evaluate Base-MLP
    base_mlp_accuracy = base_mlp.score(X_test, y_test)
    print(f"Accuracy of Base-MLP: {base_mlp_accuracy:.2f}")

    y_pred = base_mlp.predict(X_test)
    return y_pred


def top_mlp(df: pd.DataFrame, X_train, X_test, y_train, y_test):

    # Define hyperparameter grid
    param_grid = {
        'activation': ['logistic', 'tanh', 'relu'],  # Activation functions
        'hidden_layer_sizes': [(30, 50), (10, 10, 10)],  # Different network architectures
        'solver': ['adam', 'sgd']  # Solvers
        #'random_state': [36]
    }

    base_mlp = MLPClassifier()

    # Grid search and get best-performing MLP
    grid_search = GridSearchCV(base_mlp, param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    top_mlp = grid_search.best_estimator_

    # Evaluate Top-MLP
    top_mlp_accuracy = top_mlp.score(X_test, y_test)
    print(f"Accuracy of the Top-MLP: {top_mlp_accuracy:.2f}")

    y_pred = top_mlp.predict(X_test)
    return y_pred


# Calculate and append information to the results
def append_results(results, model_name, y, y_true, y_pred):

    separator = '-' * 50

    results.append(separator + "\n")

    # (A) Describe model
    results.append("(A) Model: " + model_name + "\n\n")

    # (B) Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    results.append("(B) Confusion Matrix: \n" + str(cm) + "\n\n")

    # (C) Precision, Recall, and F1-measure
    report = classification_report(y_true, y_pred, target_names=y.unique())
    results.append("(C) Precision, Recall, and F1-measure: \n" + str(report) + "\n\n")

    # (D) Accuracy, macro-average F1 and weighted-average F1 of the model
    results.append(f"(D)Accuracy, macro-average F1 and weighted-average F1 of the model\n")

    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    weighted_f1 = f1_score(y_true, y_pred, average='weighted')

    results.append(f"\t Accuracy: {accuracy:.2f}\n")
    results.append(f"\tMacro-Average F1: {macro_f1:.2f}\n")
    results.append(f"\tWeighted-Average F1: {weighted_f1:.2f}\n")

    results.append("\n\n")
    return results


def train_machines(X, y, df: pd.DataFrame, output_filename: str):

    # Split dataset into training and testing sets.
    # Add arg random_state=36 to specify random seed for reproducibility
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    y_pred_base_dt = base_dt(df, X_train, X_test, y_train, y_test, X)
    y_pred_top_dt = top_dt(df, X_train, X_test, y_train, y_test)
    y_pred_base_mlp = base_mlp(df, X_train, X_test, y_train, y_test)
    y_pred_top_mlp = top_mlp(df, X_train, X_test, y_train, y_test)

    results = []
    append_results(results, "Base-DT", y, y_test, y_pred_base_dt)
    append_results(results, "Top-DT", y, y_test, y_pred_top_dt)
    append_results(results, "Base-MLP", y, y_test, y_pred_base_mlp)
    append_results(results, "Top-MLP", y, y_test, y_pred_top_mlp)

    # Output
    with open("output/" + output_filename, "w") as file:
        file.writelines(results)


def penguins_driver():
    penguins = load_dataset_penguins('datasets/penguins.csv')
    plot(penguins, 'species', 'Species', 'output/penguins_percentages_plot.pdf')

    X = penguins.drop('species', axis=1)  # Features
    y = penguins['species']  # Target variable

    train_machines(X, y, penguins, "penguins_performance.txt")


def abalone_driver():
    abalone = load_dataset_abalone('datasets/abalone.csv')
    plot(abalone, 'Type', 'Sex', 'output/abalone_percentages_plot.pdf')

    #Split dataset
    X = abalone.drop('Type', axis=1)  # Features
    y = abalone['Type']  # Target variable

    train_machines(X, y, abalone, "abalone_performance.txt")


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    penguins_driver()
    abalone_driver()
