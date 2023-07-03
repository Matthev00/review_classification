from random import shuffle
from typing import List
from sklearn.feature_extraction.text import TfidfVectorizer


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
