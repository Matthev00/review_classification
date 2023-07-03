from data import create_dataloaders
from pathlib import Path
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from utils import test_model, save_model


def main():
    # Data preparation
    data_dir = Path("data/category")
    X_train, y_train, X_test, y_test, vectorizer = create_dataloaders(
        data_dir=data_dir)

    # Model builder
    model = svm.SVC(C=16, kernel='linear', gamma='auto')
    model.fit(X_train, y_train)

    # # Testing
    # test_set = ['great for my wedding', "loved it in my garden", 'good computer'] # noqa 5501
    # test = vectorizer.transform(test_set)

    # print(model.predict(test))

    acc, f1 = test_model(model=model,
                         X_test=X_test,
                         y_test=y_test)
    print(acc)
    print(f1)

    # Tune model
    parameters = {'kernel': ('linear', 'rbf'),
                  'C': [1, 2, 8, 16, 32]}
    svc = svm.SVC()
    model = GridSearchCV(svc, parameters, cv=5)
    model.fit(X_train, y_train)

    acc, f1 = test_model(model=model,
                         X_test=X_test,
                         y_test=y_test)
    print(acc)
    print(f1)

    # Saving moel
    save_model(model=model,
               vectorizer=vectorizer,
               model_name="SVC",
               directory="models")


if __name__ == "__main__":
    main()
