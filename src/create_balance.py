import logging
import string
import time

import spacy
from sklearn.feature_extraction.text import HashingVectorizer
from spacy.tokens import Doc

nlp = spacy.load("en_core_web_sm")


class read_and_balance:
    PUNCT_TO_REMOVE = string.punctuation

    @staticmethod
    def clean_text(doc):
        return " ".join(token.lemma_ for token in doc if not token.is_stop)

    @classmethod
    def pre_process(cls, df):
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

        logging.info(f"Processed {len(df)} rows")
        logging.info(f"Time for preprocessing a row: {end-start}")
        logging.info(
            f"Time for vectorizer: {vector_time_end - vector_time_strt}"
        )

        Y = df["category"]
        return X_trans, Y, text_preprop
