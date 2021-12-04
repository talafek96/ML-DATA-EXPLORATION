from prepare import prepare_data
from sklearn.model_selection import train_test_split
import pandas as pd


def main():
    dataset = pd.read_csv('virus_data.csv')
    train, test = train_test_split(dataset, test_size=0.2, random_state=10)
    train.reset_index(inplace=True)
    test.reset_index(inplace=True)
    print(f'Train shape is {train.shape} and test shape is {test.shape}')

    train_clean = prepare_data(train, train)
    test_clean = prepare_data(test, train)
    print(f'New train shape is {train_clean.shape} and new test shape is {test_clean.shape}')

    print(f'Train head:\n{train_clean.head(10)}\n\n')
    print(f'Test head:\n{test_clean.head(10)}\n\n')


if __name__ == '__main__':
    main()
