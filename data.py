from random import shuffle
from typing import List
from sklearn.feature_extraction.text import TfidfVectorizer
from pathlib import Path
import json
from sklearn.model_selection import train_test_split


class Category:
    ELECTRONICS = "ELECTRONICS"
    BOOKS = "BOOKS"
    CLOTHING = "CLOTHING"
    GROCERY = "GROCERY"
    PATIO = "PATIO"


class Sentiment:
    POSITIVE = "POSITIVE"
    NEGATIVE = "NEGATIVE"
    NEUTRAL = "NEUTRAL"


class Review:
    def __init__(self, category, text, score):
        self.category = category
        self.text = text
        self.score = score
        self.sentiment = self.get_sentiment()

    def get_sentiment(self):
        if self.score <= 2:
            return Sentiment.NEGATIVE
        elif self.score == 3:
            return Sentiment.NEUTRAL
        else:
            return Sentiment.POSITIVE


class ReviewContainer:
    def __init__(self, reviews: List[Review]):
        self.reviews = reviews

    def evenly_distibute(self):
        negatives = []
        neutrals = []
        positives = []
        for review in self.reviews:
            if review.sentiment == Sentiment.NEGATIVE:
                negatives.append(review)
            elif review.sentiment == Sentiment.NEUTRAL:
                neutrals.append(review)
            else:
                positives.append(review)

        len_of_samples = min(len(negatives), len(neutrals), len(positives))
        positives = positives[:len_of_samples]
        neutrals = neutrals[:len_of_samples]
        negatives = negatives[:len_of_samples]

        self.reviews = positives + neutrals + negatives
        shuffle(self.reviews)

    def get_text(self):
        return [x.text for x in self.reviews]

    def get_sentiment(self):
        return [x.sentiment for x in self.reviews]

    def get_X(self, vectorizer: TfidfVectorizer):
        return vectorizer.transform(self.get_text())

    def get_category(self):
        return [x.category for x in self.reviews]


def load_data(data_dir: Path):
    sufix = "_small.json"
    file_names = [data_dir / ("Electronics" + sufix),
                  data_dir / ("Books" + sufix),
                  data_dir / ("Clothing" + sufix),
                  data_dir / ("Grocery" + sufix),
                  data_dir / ("Patio" + sufix)]

    file_categories = [Category.ELECTRONICS,
                       Category.BOOKS,
                       Category.CLOTHING,
                       Category.GROCERY,
                       Category.PATIO]

    reviews = []
    for i in range(len(file_names)):
        file_name = file_names[i]
        category = file_categories[i]
        with open(file_name) as filehandle:
            for line in filehandle:
                review_json = json.loads(line)
                review = Review(category=category,
                                text=review_json['reviewText'],
                                score=review_json['overall'])
                reviews.append(review)

    return reviews


def create_dataloaders(data_dir: Path):

    reviews = load_data(data_dir=data_dir)
    train_dataset, test_dataset = train_test_split(reviews,
                                                   test_size=0.2,
                                                   random_state=42)

    train_container = ReviewContainer(train_dataset)
    test_container = ReviewContainer(test_dataset)

    texts = train_container.get_text()
    vectorizer = TfidfVectorizer()
    vectorizer.fit(texts)

    X_train = train_container.get_X(vectorizer=vectorizer)
    y_train = train_container.get_category()

    X_test = test_container.get_X(vectorizer=vectorizer)
    y_test = test_container.get_category()

    return X_train, y_train, X_test, y_test, vectorizer
