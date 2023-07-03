from sklearn.metrics import f1_score
from pathlib import Path
import pickle


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
    model_path = dir_path / model_name / "model.pkl"
    vect_path = dir_path / model_name / "vectorizer.pkl"

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
