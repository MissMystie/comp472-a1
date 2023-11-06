import pandas as pd
import matplotlib.pyplot as plt


def load_penguins(filepath: str) -> pd.DataFrame:

    penguins = pd.read_csv(filepath)

    #print(penguins['species'].unique())
    #print(penguins['island'].unique())
    #print(penguins['sex'].unique())

    penguins = pd.get_dummies(penguins, columns=['species', 'island'], dtype=int)
    penguins = pd.get_dummies(penguins, columns=['sex'], drop_first=True, dtype=int)

    return penguins


def load_abalone(filepath: str) -> pd.DataFrame:
    abalone = pd.read_csv(filepath)
    abalone = pd.get_dummies(abalone, columns=['Type'], dtype=int)
    return abalone


def plot_penguins(df: pd.DataFrame):
    print(df.to_string())

    #df_percent = df.groupby('sex_MALE')['body_mass_g'].value_counts(normalize=True)
    #df_percent = df_percent.mul(100).rename('Percent').reset_index()

    #print(df_percent.to_string())
    #df_percent.plot()
    #plt.show()


def plot_abalone(df: pd.DataFrame):
    print(df.to_string())

    # df_percent = df.groupby('sex_MALE')['body_mass_g'].value_counts(normalize=True)
    # df_percent = df_percent.mul(100).rename('Percent').reset_index()

    # print(df_percent.to_string())
    # df_percent.plot()
    # plt.show()


def main():
    penguins = load_penguins('penguins.csv')
    plot_penguins(penguins)
    print(penguins.to_string())

    abalone = load_abalone('abalone.csv')
    plot_abalone(abalone)
    print(abalone.to_string())


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
