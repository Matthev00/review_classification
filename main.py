from data import create_dataloaders
from pathlib import Path
from sklearn import svm
from sklearn.metrics import f1_score


def main():
    # Data preparation
    data_dir = Path("data/category")
    X_train, y_train, X_test, y_test, vectorizer = create_dataloaders(
        data_dir=data_dir)

    # Model builder
    model = svm.SVC(C=16, kernel='linear', gamma='auto')
    model.fit(X_train, y_train)

    # test_set = ['great for my wedding', "loved it in my garden", 'good computer']
    # test = vectorizer.transform(test_set)

    # print(model.predict(test))
    print(model.score(X_test, y_test))
    y_pred = model.predict(X_test)

    f1_result = f1_score(y_test, y_pred, average=None)
    print(f1_result)





if __name__ == "__main__":
    main()
