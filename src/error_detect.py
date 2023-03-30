import logging
import os
import uuid

import pandas as pd
from tqdm import tqdm

from classification import classify

logging.basicConfig(
    filename="logs" + os.sep + "error_detection.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

console = logging.StreamHandler()
console.setLevel(logging.INFO)
logging.getLogger().addHandler(console)


class error_detect:
    INPUT_FOLDER = "." + os.sep + "data" + os.sep
    OUTPUT_FOLDER = "./out_data"
    OUTPUT_FILE = str(uuid.uuid1()).replace("-", "_") + ".csv"

    @classmethod
    def find_files(cls, folder):
        """
        This method returns a list of file paths with .csv extension in the given folder.

        Args:
        - folder (str): A string representing the path to the folder where files are located.

        Returns:
        - A list of file paths with .csv extension.
        """
        return [
            os.path.join(folder, f)
            for f in os.listdir(folder)
            if f.endswith(".csv")
        ]

    @classmethod
    def create_data_frame(cls):
        """
        This method creates a pandas DataFrame by combining the data from all the CSV files in the INPUT_FOLDER.

        Args:
        - None

        Returns:
        - A pandas DataFrame that contains the combined data from all the CSV files in the INPUT_FOLDER.
        """
        data = error_detect.find_files(cls.INPUT_FOLDER)
        logging.info(f"Found {len(data)} input files.")
        ext_data = [pd.read_csv(f, encoding="ISO-8859-1") for f in data]
        df = pd.concat(ext_data, ignore_index=True)
        logging.info(f"Combined {len(df)} rows of data.")
        return df

    @classmethod
    def detect(cls):
        """
        This method detects errors in the data and generates predictions for the labels of the data.

        Args:
        - None

        Returns:
        - None
        """
        try:
            df = error_detect.create_data_frame()
            df.dropna(inplace=True)
            df.reset_index(drop=True, inplace=True)
            # df = df.head(75) Only to use a small subset of the dataframe
            predictions = []
            logging.debug(f"Processing {len(df)} rows of data.")
            for i in tqdm(range(len(df))):
                t_df = df.drop(index=i)
                actual_cat = df.loc[i, "category"]
                model, vectorizer = classify.label_detect(t_df)
                i_trans = vectorizer.transform([df.loc[i, "text"]])
                pred_l = model.predict(i_trans)[0]
                pred_probs = model.predict_proba(i_trans)[0].max()
                predictions.append(
                    {
                        "predicted_label": pred_l,
                        "probability": round(pred_probs, 2),
                    }
                )
                logging.info(
                    f"Row {i} - actual label: {actual_cat}, predicted label: {pred_l}, probability: {pred_probs}"
                )
            df = pd.concat([df, pd.DataFrame(predictions)], axis=1)
            output_file_path = os.path.join(cls.OUTPUT_FOLDER, cls.OUTPUT_FILE)
            df.to_csv(output_file_path, index=False)
            logging.info(f"Saved output to {output_file_path}")
        except Exception as e:
            logging.error(f"Error: {str(e)}")


if __name__ == "__main__":
    error_detect.detect()
