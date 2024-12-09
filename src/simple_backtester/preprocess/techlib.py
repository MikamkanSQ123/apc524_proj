import yaml
from abc import ABC, abstractmethod
from typing import final, Any, Union
from pathlib import Path
import pandas as pd
import numpy as np
from numpy.typing import NDArray


class Feature(ABC):
    "Abstract class for self-designed features (may also include simple features)"

    @final
    def __init__(self, config_path: Union[str, Path]):
        if isinstance(config_path, str):
            config_path = Path(config_path)
        config: dict[str, Any] = yaml.safe_load(config_path.read_text())

        required_sections = {"feature_parameters"}
        missing_sections = required_sections - config.keys()
        if missing_sections:
            raise ValueError(f"Missing mandatory sections in YAML: {missing_sections}")

        self.__freq = config["setup"]["frequency"]
        self.__featurparams = config["feature_parameters"]

    @abstractmethod
    def eval(
        self, data: Union[NDArray[np.float64], "pd.Series[Any]", pd.DataFrame]
    ) -> Any:
        "Some evaluation function"
        self.func = lambda x: np.exp(x)
        self.feature: Any = self.func(data)  # type: ignore[no-untyped-call]
        return self.feature
