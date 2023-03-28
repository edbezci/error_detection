import logging
import warnings

from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, train_test_split

from create_balance import read_and_balance

warnings.filterwarnings("ignore")


class classify:
    parameters = {
        "loss": ["log_loss"],
        "penalty": ("l2", "elasticnet"),
        "tol": (1e-2, 1e-3),
    }

    clf = SGDClassifier()

    @classmethod
    def label_detect(cls, df):
        X, Y, vectorizer = read_and_balance.pre_process(df)

        X_train, X_test, y_train, y_test = train_test_split(
            X, Y, test_size=0.33, random_state=42
        )
        gs_clf = GridSearchCV(cls.clf, cls.parameters, cv=5, n_jobs=-1).fit(
            X_train, y_train
        )

        # Configure the logger
        logging.basicConfig(
            filename="error_detection.log",
            filemode="a",
            format="%(asctime)s - %(levelname)s - %(message)s",
        )
        logger = logging.getLogger()

        logger.info("Optimal Parametres for this classification job:")
        for param_name in sorted(cls.parameters.keys()):
            logger.info("%s: %r", param_name, gs_clf.best_params_[param_name])

        preds = gs_clf.predict(X_test)
        cr = classification_report(
            y_test, preds, target_names=["Safe", "Scam"]
        )
        logger.info(cr)

        return gs_clf, vectorizer
