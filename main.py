import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.metrics import confusion_matrix, classification_report, f1_score, accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split, GridSearchCV
import numpy as np


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


def plot(df: pd.DataFrame, target: str, label_x: str, output_filepath: str, animal: str):
    # Calculate the percentage of instances for the output class
    class_percent = df[target].value_counts(normalize=True) * 100

    # Create a bar plot
    plt.figure(figsize=(8, 6))
    class_percent.plot(kind='bar', color='skyblue')
    plt.title(f'Percentage of instances for each output class for {animal}')
    plt.xlabel(label_x)
    plt.ylabel('Percentage')
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Save the plot in the PDF file
    with PdfPages(output_filepath) as pdf:
        pdf.savefig()

    # Show the plot
    plt.show()
    plt.close()


def base_dt(animal, X_train, X_test, y_train, y_test, X, max_depth, output):
    # Train DT classifier with default parameters
    base_dt = DecisionTreeClassifier(max_depth=max_depth)
    base_dt.fit(X_train, y_train)

    base_dt_accuracy = base_dt.score(X_test, y_test)
    print(f"Accuracy of Base-DT for {animal}: {base_dt_accuracy:.2f}")

    y_pred = base_dt.predict(X_test)

    # Visualize
    plt.figure(figsize=(15, 10))
    plot_tree(base_dt, feature_names=X.columns, class_names=base_dt.classes_, filled=True, max_depth=max_depth)

    # Save the decision tree plot in the PDF file
    output.savefig()
    plt.show()

    # Show the plot
    plt.close()

    return y_pred


def top_dt(animal, X_train, X_test, y_train, y_test, max_depth=None):
    # Define hyperparameter grid
    param_grid = {
        'criterion': ['gini', 'entropy'],
        'max_depth': [None, 5, 10, max_depth],
        'min_samples_split': [2, 5, 10]
    }

    base_dt = DecisionTreeClassifier()

    # Grid search and get best-performing DT
    grid_search = GridSearchCV(base_dt, param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)

    # Evaluate Top-DT
    top_dt = grid_search.best_estimator_
    top_dt_accuracy = top_dt.score(X_test, y_test)
    print(f"Accuracy of Top-DT for {animal}: {top_dt_accuracy:.2f}")

    y_pred = top_dt.predict(X_test)

    return y_pred


def base_mlp(animal, X_train, X_test, y_train, y_test):
    base_mlp = MLPClassifier(
        hidden_layer_sizes=(100, 100),  # 2 hidden layers with 100 neurons each
        activation='logistic',  # Sigmoid (logistic) activation function
        solver='sgd',  # Stochastic gradient descent
        #random_state=36,  # For reproducibility
        max_iter=1200,
        learning_rate_init=0.001
    )

    base_mlp.fit(X_train, y_train)

    # Evaluate Base-MLP
    base_mlp_accuracy = base_mlp.score(X_test, y_test)
    print(f"Accuracy of Base-MLP for {animal}: {base_mlp_accuracy:.2f}")

    y_pred = base_mlp.predict(X_test)
    return y_pred


def top_mlp(animal, X_train, X_test, y_train, y_test):
    # Define hyperparameter grid
    param_grid = {
        'activation': ['logistic', 'tanh', 'relu'],  # Activation functions
        'hidden_layer_sizes': [(30, 50), (10, 10, 10)],  # Different network architectures
        'solver': ['adam', 'sgd']  # Solvers
        # 'random_state': [36]
    }

    base_mlp = MLPClassifier(max_iter=1000, early_stopping=True, validation_fraction=0.1)

    # Grid search and get best-performing MLP
    grid_search = GridSearchCV(base_mlp, param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    top_mlp = grid_search.best_estimator_

    # Evaluate Top-MLP
    top_mlp_accuracy = top_mlp.score(X_test, y_test)
    print(f"Accuracy of the Top-MLP for {animal}: {top_mlp_accuracy:.2f}")

    y_pred = top_mlp.predict(X_test)
    return y_pred


def append_results(results, model_name, y, y_true, y_pred, run):
    separator = '-' * 50
    results.append(separator + "\n")
    results.append(f"Run {run} - (A) Model: {model_name}\n\n")
    cm = confusion_matrix(y_true, y_pred)
    results.append("(B) Confusion Matrix: \n" + str(cm) + "\n\n")
    report = classification_report(y_true, y_pred, target_names=y.unique(), zero_division=1)
    results.append("(C) Precision, Recall, and F1-measure: \n" + str(report) + "\n\n")
    results.append(f"(D) Accuracy, macro-average F1 and weighted-average F1 of the model\n")
    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    weighted_f1 = f1_score(y_true, y_pred, average='weighted')
    results.append(f"\tAccuracy: {accuracy:.2f}\n")
    results.append(f"\tMacro-Average F1: {macro_f1:.2f}\n")
    results.append(f"\tWeighted-Average F1: {weighted_f1:.2f}\n")
    results.append("\n\n")
    return results, accuracy, macro_f1, weighted_f1


def train_machines(animal, X, y, output_filepath: str, runs: int = 1, max_depth=None):
    accuracies = []
    macro_f1s = []
    weighted_f1s = []

    decision_tree_filepath = f'output/{animal}_decision_trees.pdf'

    with PdfPages(decision_tree_filepath) as dt_output:
        for run in range(runs):
            # Split dataset into training and testing sets.
            # Add arg random_state=36 to specify random seed for reproducibility
            X_train, X_test, y_train, y_test = train_test_split(X, y)

            print(f'= Dataset: {animal}, run {run + 1} =')

            y_pred_base_dt = base_dt(animal, X_train, X_test, y_train, y_test, X, max_depth, dt_output)
            y_pred_top_dt = top_dt(animal, X_train, X_test, y_train, y_test, max_depth)
            y_pred_base_mlp = base_mlp(animal, X_train, X_test, y_train, y_test)
            y_pred_top_mlp = top_mlp(animal, X_train, X_test, y_train, y_test)

            print("\n")

            results = []
            results, acc_base_dt, macro_f1_base_dt, weighted_f1_base_dt = append_results(results, "Base-DT", y, y_test,
                                                                                         y_pred_base_dt, run)
            results, acc_top_dt, macro_f1_top_dt, weighted_f1_top_dt = append_results(results, "Top-DT", y, y_test,
                                                                                      y_pred_top_dt, run)
            results, acc_base_mlp, macro_f1_base_mlp, weighted_f1_base_mlp = append_results(results, "Base-MLP", y,
                                                                                            y_test, y_pred_base_mlp,
                                                                                            run)
            results, acc_top_mlp, macro_f1_top_mlp, weighted_f1_top_mlp = append_results(results, "Top-MLP", y, y_test,
                                                                                         y_pred_top_mlp, run)

            # Output
            with open(output_filepath, "a") as file:
                file.writelines(results)

            accuracies.append([acc_base_dt, acc_top_dt, acc_base_mlp, acc_top_mlp])
            macro_f1s.append([macro_f1_base_dt, macro_f1_top_dt, macro_f1_base_mlp, macro_f1_top_mlp])
            weighted_f1s.append([weighted_f1_base_dt, weighted_f1_top_dt, weighted_f1_base_mlp, weighted_f1_top_mlp])

    return accuracies, macro_f1s, weighted_f1s


def main():
    animals = ['penguins', 'abalone']

    all_accuracies = []
    all_macro_f1s = []
    all_weighted_f1s = []

    for animal in animals:

        dataset_filepath = f'datasets/{animal}.csv'
        percentages_plot_filepath = f'output/{animal}_percentages_plot.pdf'
        performance_filepath = f'output/{animal}_performance.txt'

        if animal == 'penguins':
            dataset = load_dataset_penguins(dataset_filepath)
            target_var, target_label = 'species', 'Species'
        elif animal == 'abalone':
            dataset = load_dataset_abalone(dataset_filepath)
            target_var, target_label = 'Type', 'Sex'


        plot(dataset, target_var, target_label, percentages_plot_filepath, animal)  # Save the bar graph
        X = dataset.drop(target_var, axis=1)  # Features
        y = dataset[target_var]  # Target variable

        accuracies, macro_f1s, weighted_f1s = train_machines(animal, X, y, performance_filepath, 5, 4)

        all_accuracies.append(accuracies)
        all_macro_f1s.append(macro_f1s)
        all_weighted_f1s.append(weighted_f1s)

    # Calculate average and variance
    avg_accuracies = np.mean(all_accuracies, axis=0)
    var_accuracies = np.var(all_accuracies, axis=0)
    avg_macro_f1s = np.mean(all_macro_f1s, axis=0)
    var_macro_f1s = np.var(all_macro_f1s, axis=0)
    avg_weighted_f1s = np.mean(all_weighted_f1s, axis=0)
    var_weighted_f1s = np.var(all_weighted_f1s, axis=0)

    # Display results
    for i, animal in enumerate(animals):
        #Redirect standard output to the file
        with open(f"output/{animal}_performance.txt", "a") as output_file:
            output_file.writelines(f"\n{animal.capitalize()} Results:")
            print(f"\n{animal.capitalize()} Results:")
            for j, model in enumerate(["Base-DT", "Top-DT", "Base-MLP", "Top-MLP"]):
                output_file.writelines(f"\n{model} Results: \n")
                print(f"\n{model} Results:")
                output_file.writelines(f"Average Accuracy: {avg_accuracies[i, j]:.2f}, Variance: {var_accuracies[i, j]:.2f}\n")
                print(f"Average Accuracy: {avg_accuracies[i, j]:.2f}, Variance: {var_accuracies[i, j]:.2f}")
                output_file.writelines(f"Average Macro F1: {avg_macro_f1s[i, j]:.2f}, Variance: {var_macro_f1s[i, j]:.2f}\n")
                print(f"Average Macro F1: {avg_macro_f1s[i, j]:.2f}, Variance: {var_macro_f1s[i, j]:.2f}")
                output_file.writelines(f"Average Weighted F1: {avg_weighted_f1s[i, j]:.2f}, Variance: {var_weighted_f1s[i, j]:.2f}\n")
                print(f"Average Weighted F1: {avg_weighted_f1s[i, j]:.2f}, Variance: {var_weighted_f1s[i, j]:.2f}")


if __name__ == '__main__':
    main()
