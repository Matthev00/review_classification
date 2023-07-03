from sklearn.metrics import f1_score


def test_model(model, X_test, y_test):
    accuracy = model.score(X_test, y_test)

    y_pred = model.predict(X_test)
    f1_result = f1_score(y_test, y_pred, average=None)

    return accuracy, f1_result