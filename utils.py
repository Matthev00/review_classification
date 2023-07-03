from sklearn.metrics import f1_score, confusion_matrix
from pathlib import Path
import pickle
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
from data import Category


def test_model(model, X_test, y_test):
    accuracy = model.score(X_test, y_test)

    y_pred = model.predict(X_test)
    f1_result = f1_score(y_test, y_pred, average=None)

    return accuracy, f1_result


def save_model(model,
               vectorizer,
               model_name: str,
               directory: str):

    dir_path = Path(directory)
    model_dir = dir_path / model_name
    model_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / "model.pkl"
    vect_path = model_dir / "vectorizer.pkl"

    with open(model_path, 'wb') as f:
        pickle.dump(model, f)

    with open(vect_path, 'wb') as f:
        pickle.dump(vectorizer, f)


def load_model(path: Path):
    model_path = path / "model.pkl"
    vect_path = path / "vectorizer.pkl"

    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    with open(vect_path, 'rb') as f:
        vectorizer = pickle.load(f)

    return model, vectorizer


def plot_confusion_matrix(model,
                          X_test,
                          y_test):

    y_pred = model.predict(X_test)

    labels = [Category.ELECTRONICS,
              Category.BOOKS,
              Category.CLOTHING,
              Category.GROCERY,
              Category.PATIO]

    conf_matrix = confusion_matrix(y_true=y_test,
                                   y_pred=y_pred,
                                   labels=labels)
    df_confusion_matrix = pd.DataFrame(conf_matrix,
                                       index=labels,
                                       columns=labels)

    sn.heatmap(df_confusion_matrix, annot=True, fmt='d')
    plt.show()
