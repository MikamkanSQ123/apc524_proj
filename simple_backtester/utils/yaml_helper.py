import yaml
from typing import Any, Dict


class YamlParser:
    """
    A class to parse and save YAML files.

    Attributes:
        file_path (str): The path to the YAML file.

    Methods:
        load_yaml() -> Dict[str, Any]:
            Loads and returns the contents of the YAML file as a dictionary.

        save_yaml(data: Dict[str, Any]) -> None:
            Saves the given dictionary data to the YAML file.
    """

    def __init__(self, file_path: str) -> None:
        self.file_path = file_path

    def load_yaml(self) -> Dict[str, Any]:
        with open(self.file_path, "r") as file:
            data: Dict[str, Any] = yaml.safe_load(file)
        return data

    def save_yaml(self, data: Dict[str, Any]) -> None:
        with open(self.file_path, "w") as file:
            yaml.safe_dump(data, file)
