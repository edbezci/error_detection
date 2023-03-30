import logging
import string
import time

import spacy
from sklearn.feature_extraction.text import HashingVectorizer

nlp = spacy.load("en_core_web_sm")


class read_and_balance:
    """
    A class to read and balance text data.

    ...

    Attributes
    ----------
    PUNCT_TO_REMOVE : str
        string of punctuation characters to remove from text

    Methods
    -------
    clean_text(doc):
        Removes stop words and lemmatizes a given spacy doc object.

    pre_process(df):
        Preprocesses the text data in a given pandas DataFrame object, including
        dropping missing and duplicate rows, cleaning the text, and vectorizing
        the text data using the HashingVectorizer.

    """

    PUNCT_TO_REMOVE = string.punctuation

    @staticmethod
    def clean_text(doc):
        """
        Removes stop words and lemmatizes a given spacy doc object.

        Parameters
        ----------
        doc : spacy.tokens.doc.Doc
            A spacy doc object representing a single document of text.

        Returns
        -------
        str
            A string of lemmatized text with stop words removed.
        """
        return " ".join(token.lemma_ for token in doc if not token.is_stop)

    @classmethod
    def pre_process(cls, df):
        """
        Preprocesses the text data in a given pandas DataFrame object, including
        dropping missing and duplicate rows, cleaning the text, and vectorizing
        the text data using the HashingVectorizer.

        Parameters
        ----------
        df : pandas.DataFrame
            A pandas DataFrame object representing the text data to preprocess.

        Returns
        -------
        tuple
            A tuple containing the vectorized text data, labels, and HashingVectorizer object.
        """
        df = df.dropna(axis=0, how="any")
        df = df.drop_duplicates(subset=["text"], keep="first")
        start = time.time()

        docs = list(nlp.pipe(df["text"], batch_size=2000))
        df["clean_text"] = [cls.clean_text(doc) for doc in docs]

        end = time.time()

        vector_time_strt = time.time()
        text_preprop = HashingVectorizer(ngram_range=(1, 1))
        X_trans = text_preprop.fit_transform(df["clean_text"])
        vector_time_end = time.time()

        logging.basicConfig(
            filename="error_detection.log",
            filemode="a",
            format="%(asctime)s - %(levelname)s - %(message)s",
        )
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)

        logging.info(f" Processed {len(df)} rows")
        logging.info(f"Time for preprocessing a row: {end-start}")
        logging.info(
            f"Time for vectorizer: {vector_time_end - vector_time_strt}"
        )

        Y = df["category"]
        return X_trans, Y, text_preprop
