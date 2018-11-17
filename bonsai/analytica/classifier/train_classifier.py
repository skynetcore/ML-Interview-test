'''
/*
 * Copyright (C) 2018 Derric Lyns [derriclyns@gmail.com]
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
'''

# Data Manipulation Libraries
import collections as c
import sqlite3
# NLP Library
from nltk.classify.naivebayes import NaiveBayesClassifier
# SVM Library
from sklearn import svm
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.cluster import KMeans
from sklearn.linear_model import Lasso

import pandas as pd

# Argument Parsing
import argparse
# Pretty Print
import pprint


# Current genre schema
GENRES = {1: 'Fantasy', 2: 'YA', 3: 'Urban Fantasy', 4: 'Sci-Fi',
          5: 'Horror/Thriller/Mystery', 6: 'Historical Fiction',
          7: 'Nonfiction'}

GENRES_LIST = ['Fantasy', 'YA', 'Urban Fantasy', 'Sci-Fi',
               'Horror/Thriller/Mystery', 'Historical Fiction',
               'Nonfiction']


def parse_args():
    """
    Parses input arguments.

    :return: An argparse object.
    """
    parser = argparse.ArgumentParser(
        description="A script for training classifiers on Goodreads data.")
    parser.add_argument('--clf_type', default="svm", nargs="?",
                        const='1', help="The type of classifier used.")
    parser.add_argument('--random_state', default="42", nargs="?",
                        const='1', help="An integer value for random seeding.")
    parser.add_argument('--database_name', default="classifier_books.db",
                        nargs="?", const='1', help="The name of the database "
                                                   "to be accessed.")
    arguments = parser.parse_args()
    return arguments


def title_prep(title_string):
    """
    Break titles into lists of words

    :param title_string:
    :return:
    """
    return c.Counter(title_string.split())


def author_prep(author_string):
    """
    Combine author names into a single 'word',
    since they aren't much use as first name / last name.

    :param author_string:
    :return:
    """
    return c.Counter([author_string.replace(' ', '_')])

def generate_panda_test_train(pd, RANDOM_STATE):
    """
    Pull data from the associated database and split into test/train.

    :param cur: Database pointer.
    :return:
    """
    
    '''
    cur.execute(
        "SELECT indexer, isbn, title, author, genre_class, num_pages,"
        "publication_year from Books "
        "ORDER BY isbn ASC")
    x = cur.fetchall()
    '''

    features = list([pd["title"], pd["author"]])
    labels = list(pd["genre"])

    features_train, features_test, labels_train, labels_test = \
        train_test_split(features, labels, test_size=0.2,
                         random_state=RANDOM_STATE)

    return features_train, features_test, labels_train, labels_test

def generate_test_train(cur, RANDOM_STATE):
    """
    Pull data from the associated database and split into test/train.

    :param cur: Database pointer.
    :return:
    """

    cur.execute(
        "SELECT indexer, isbn, title, author, genre_class, num_pages,"
        "publication_year from Books "
        "ORDER BY isbn ASC")
    x = cur.fetchall()

    features = [(record[2], record[3], record[5], record[6]) for record in x]
    labels = [(record[4]) for record in x]

    features_train, features_test, labels_train, labels_test = \
        train_test_split(features, labels, test_size=0.2,
                         random_state=RANDOM_STATE)

    return features_train, features_test, labels_train, labels_test


class ClassifierComposite:
    """
    A class that encapsulates the primary classifier and the nlp classifiers.
    Allows for multiple types of classifiers.
    """
    def __init__(self):
        self.clf = None
        self.nlp_title = None
        self.nlp_author = None

    def tune_features(self, features_train, labels_train):
        """
        Assess feature importance.

        :param features_train:
        :param labels_train:
        :return:
        """
        clf = Lasso(alpha=0.1)
        clf.fit(features_train, labels_train)
        fv = list(zip(GENRES, clf.coef_))
        pprint.pprint(fv)

    def train_nlp(self, features_train, labels_train):
        """
        Trains two Naive Bayes classifiers, one for author names
        and one for titles.

        :param cur: A database pointer for the Goodreads data (see pull_data.py).
        :param con: The database connection.
        :return: nlp_title: An NLP classifier for titles.
                 nlp_author: An NLP classifier for authors.
        """
        # Organize title and author data
        train_data = list(zip(features_train, labels_train))

        title_data = [(title_prep(record[0][0]), record[1]) for record in
                      train_data]

        author_data = [(author_prep(record[0][1]), record[1]) for
                       record in train_data]

        # Train the classifiers using the training data

        # Title Classifier
        self.nlp_title = NaiveBayesClassifier.train(title_data)
        # nltk.classify.util.accuracy(clf, title_data[divide:])

        # Author Classifier
        self.nlp_author = NaiveBayesClassifier.train(author_data)
        # nltk.classify.util.accuracy(clf2, author_data[divide:])

    def train_kmeans(self, features_train, features_test, labels_train,
                     labels_test, RANDOM_STATE):
        """
        Train a clustered kmeans classifier.

        :param features_train:
        :param features_test:
        :param labels_train:
        :param labels_test:
        :param RANDOM_STATE:
        :return:
        """
        self.train_nlp(features_train, labels_train)

        features_train = [(self.nlp_title.classify(title_prep(record[0])),
                           self.nlp_author.classify(author_prep(record[1])))
                          for record in features_train]

        features_test = [(self.nlp_title.classify(title_prep(record[0])),
                          self.nlp_author.classify(author_prep(record[1])))
                         for record in features_test]

        self.clf = KMeans(n_clusters=7, random_state=RANDOM_STATE)
        self.clf.fit(features_train, labels_train)

        pred = self.clf.predict(features_test)
        return classification_report(labels_test, pred,
                                     target_names=GENRES_LIST)

    def train_svm_minimum(self, features_train, features_test, labels_train,
                          labels_test, RANDOM_STATE):
        """
        Train an SVM on title and author information.

        :param features_train:
        :param features_test:
        :param labels_train:
        :param labels_test:
        :return:
        """
        self.train_nlp(features_train, labels_train)

        features_train = [(self.nlp_title.classify(title_prep(record[0])),
                           self.nlp_author.classify(author_prep(record[1])))
                          for record in features_train]

        features_test = [(self.nlp_title.classify(title_prep(record[0])),
                          self.nlp_author.classify(author_prep(record[1])))
                         for record in features_test]

        self.clf = svm.SVC(kernel="rbf", C=1.0, random_state=RANDOM_STATE)
        self.clf.fit(features_train, labels_train)

        pred = self.clf.predict(features_test)
        return classification_report(labels_test, pred,
                                     target_names=GENRES_LIST)

    def train_svm_full(self, features_train, features_test, labels_train,
                       labels_test, RANDOM_STATE):
        """
        Train an SVM on several features.

        So far;
        Title Class, Title Probability, Author Class, Number of Pages,
        Publication Year

        :param features_train:
        :param features_test:
        :param labels_train:
        :param labels_test:
        :return:
        """
        self.train_nlp(features_train, labels_train)

        def title_class(title_string):
            return self.nlp_title.classify(title_prep(title_string))

        def prob_of_title_class(title_string):
            return max(self.nlp_title.prob_classify(
                title_prep(title_string[0]))._prob_dict.values())

        # def second_title_class(title_string):
        #     cp = copy.deepcopy(self.nlp_title.prob_classify(
        #         title_prep(title_string[0])))
        #
        #     pass

        def null_fixer(epsilon):
            return 0

        def author_class(author_string):
            return self.nlp_author.classify(author_prep(author_string))

        features_train = [(title_class(record[0]),
                           prob_of_title_class(record[0]),
                           author_class(record[1]),
                           null_fixer(record[2]),
                           null_fixer(record[3]))
                          for record in features_train]

        features_test = [(title_class(record[0]),
                           prob_of_title_class(record[0]),
                           author_class(record[1]),
                          null_fixer(record[2]),
                          null_fixer(record[3]))
                          for record in features_test]

        self.clf = svm.SVC(kernel="rbf", C=1.0, random_state=RANDOM_STATE)
        self.clf.fit(features_train, labels_train)

        pred = self.clf.predict(features_test)
        return classification_report(labels_test, pred,
                                     target_names=GENRES_LIST)

    def train_ada(self, features_train, features_test, labels_train,
                  labels_test, RANDOM_STATE):
        """
        Train an AdaBoost classifier.

        :param features_train:
        :param features_test:
        :param labels_train:
        :param labels_test:
        :param RANDOM_STATE:
        :return:
        """
        self.train_nlp(features_train, labels_train)

        def title_class(title_string):
            return self.nlp_title.classify(title_prep(title_string))

        def prob_of_title_class(title_string):
            return max(self.nlp_title.prob_classify(
                title_prep(title_string[0]))._prob_dict.values())

        def null_fixer(epsilon):
            return 0

        def author_class(author_string):
            return self.nlp_author.classify(author_prep(author_string))

        features_train = [(title_class(record[0]),
                           prob_of_title_class(record[0]),
                           author_class(record[1]),
                           null_fixer(record[2]),
                           null_fixer(record[3]))
                          for record in features_train]

        features_test = [(title_class(record[0]),
                           prob_of_title_class(record[0]),
                           author_class(record[1]),
                          null_fixer(record[2]),
                          null_fixer(record[3]))
                          for record in features_test]

        self.clf = AdaBoostClassifier()
        self.clf.fit(features_train, labels_train)

        pred = self.clf.predict(features_test)
        return classification_report(labels_test, pred,
                                     target_names=GENRES_LIST)


if __name__ == '__main__':
    # Parse input arguments
    args = parse_args()
    RANDOM_STATE = int(args.random_state)
    DATABASE_NAME = args.database_name
    CLF_TYPE = args.clf_type

    # Connect to SQLite Database
    try:
        con = sqlite3.connect(DATABASE_NAME)
        cur = con.cursor()
        print('')

        # Pull data for testing and training
        features_train, features_test, labels_train, labels_test = \
            generate_test_train(cur, RANDOM_STATE)

        # Create classifier
        clf_composite = ClassifierComposite()

        if CLF_TYPE == 'svm':
            out = clf_composite.train_svm_minimum(features_train, features_test,
                                                  labels_train, labels_test,
                                                  RANDOM_STATE)
        elif CLF_TYPE == 'svm_full':
            out = clf_composite.train_svm_full(features_train, features_test,
                                               labels_train, labels_test,
                                               RANDOM_STATE)
        elif CLF_TYPE == 'ada':
            out = clf_composite.train_ada(features_train, features_test,
                                          labels_train, labels_test,
                                          RANDOM_STATE)
        elif CLF_TYPE == 'kmeans':
            out = clf_composite.train_kmeans(features_train, features_test,
                                             labels_train, labels_test,
                                             RANDOM_STATE)
        else:
            out = 'Unrecognized "clf_type" argument.'

        # Output classification report
        # clf_composite.tune_features(features_train, labels_train)
        print(out)

    except sqlite3.OperationalError:
        print('Could not access database: ' + DATABASE_NAME)






