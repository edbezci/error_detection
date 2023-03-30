Error Detection and Classification

This codebase detects errors in data and generates predictions for the labels of the data using machine learning. The code uses the Poetry package manager for dependency management.

Installation

    Clone the repository: git clone https://github.com/edbezci/error-detection.git
    Navigate into the repository: cd error-detection
    Install dependencies: poetry install
    Activate poetry: poetry shell

Usage

    Place the input data CSV files in the data folder.
    Navigate into 'src': cd src
    Run the code: python error_detect.py
    The output will be generated in the out_data folder as a CSV file with a unique file name.

Logging

The code uses the logging module to log events and errors to a log file and the console.

Testing

The code uses pytest for unit testing.

    Navigate into 'src': cd src
    Run the code: pytest ..\test

License

This code is released under the MIT License.
