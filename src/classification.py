import logging
import warnings

from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, train_test_split

from create_balance import read_and_balance

warnings.filterwarnings("ignore")


class classify:
    """
    A class to perform text classification using a linear classifier with grid search for hyperparameter tuning.

    Attributes:
    -----------
    parameters : dict
        A dictionary containing the hyperparameters to be tuned by grid search.
    clf : SGDClassifier
        A linear classifier object from Scikit-learn's SGDClassifier class.

    Methods:
    --------
    label_detect(df):
        Trains a linear classifier on a balanced dataset obtained from pre-processing input DataFrame `df` using
        the `read_and_balance` module, and returns the trained classifier and the corresponding fitted vectorizer.
    """

    parameters = {
        "loss": ["log_loss"],
        "penalty": ("l2", "elasticnet"),
        "tol": (1e-2, 1e-3),
    }

    clf = SGDClassifier()

    @classmethod
    def label_detect(cls, df):
        """
        Trains a linear classifier on a balanced dataset obtained from pre-processing input DataFrame `df`
        using the `read_and_balance` module, and returns the trained classifier and the corresponding fitted vectorizer.

        Parameters:
        -----------
        df : pandas.DataFrame
            A DataFrame containing text data to be classified, with columns "text" and "category".

        Returns:
        --------
        gs_clf : GridSearchCV
            A grid search object that contains the best hyperparameters found during training.
        vectorizer : TfidfVectorizer
            A TfidfVectorizer object that was fitted on the training data.
        """
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
