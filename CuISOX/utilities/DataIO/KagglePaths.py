from CuISOX.utilities.configure_paths import DataPaths
from pathlib import Path

class KagglePaths:

    _SUBDIRPATH = "sub_dir_path"
    _DATAFILES = "datafiles"

    def __init__(self):
        
        self.data = {
            "DigitRecognizer": 
            {
                "sub_dir_path": "DigitRecognizer",
                "datafiles": [
                    {
                        "filename": "sample_submission.csv",
                        "sub_dir_path": "digit-recognizer/sample_submission.csv"
                    },
                    {
                        "filename": "test.csv",
                        "sub_dir_path": "digit-recognizer/test.csv"
                    },
                    {
                        "filename": "train.csv",
                        "sub_dir_path": "digit-recognizer/train.csv"
                    }
                ]
            }
        }

    def get_all_data_file_paths(self):
        result = {}
        for key in self.data:
            # In Python, this is a reference to the object, not a deep copy of
            # it. This means changes made through this new variable will reflect
            # in the original dictionary.
            element = self.data[key]
            base_path = Path(element[KagglePaths._SUBDIRPATH])
            possible_data_file_paths = []
            if KagglePaths._DATAFILES in element:
                for data_file in element[KagglePaths._DATAFILES]:
                    possible_data_file_paths.append(
                        base_path / Path(data_file[KagglePaths._SUBDIRPATH]))
            result[key] = possible_data_file_paths
        return result
                                                    