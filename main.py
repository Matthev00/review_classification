from data import create_dataloaders
from pathlib import Path


def main():
    data_dir = Path("data/category")
    X_train, y_train, X_test, y_test = create_dataloaders(data_dir=data_dir)
    print(X_train.shape)
    print(len(y_train))


if __name__ == "__main__":
    main()
