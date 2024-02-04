import os
import warnings
from datetime import datetime
from typing import List, Tuple, Union

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")


class DelayModel:

    def __init__(self) -> None:
        self._save_dir = "models"
        if not os.path.exists(self._save_dir):
            os.makedirs(self._save_dir)

        self._model = xgb.XGBClassifier()
        self._model.load_model(f"{self._save_dir}/model.json")

        self._feature_columns = [
            "OPERA_Latin American Wings",
            "MES_7",
            "MES_10",
            "OPERA_Grupo LATAM",
            "MES_12",
            "TIPOVUELO_I",
            "MES_4",
            "MES_11",
            "OPERA_Sky Airline",
            "OPERA_Copa Air",
        ]

    def preprocess(
        self, data: pd.DataFrame, target_column: str | None = None
    ) -> Union[Tuple[pd.DataFrame, pd.DataFrame], pd.DataFrame]:
        """
        Prepare raw data for training or predict.

        Args:
            data (pd.DataFrame): raw data.
            target_column (str, optional): if set, the target is returned.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: features and target.
            or
            pd.DataFrame: features.
        """
        data["min_diff"] = data.apply(self._get_min_diff, axis=1)

        THRESHOLD_IN_MINUTES = 15
        data["delay"] = np.where(data["min_diff"] > THRESHOLD_IN_MINUTES, 1, 0)

        features = pd.concat(
            [
                pd.get_dummies(data["OPERA"], prefix="OPERA"),
                pd.get_dummies(data["TIPOVUELO"], prefix="TIPOVUELO"),
                pd.get_dummies(data["MES"], prefix="MES"),
            ],
            axis=1,
        )

        features = pd.DataFrame(data, columns=self._feature_columns)

        if target_column is not None:
            target = data[[target_column]]
            return features, target

        return features

    def _get_min_diff(self, data: pd.DataFrame) -> float | None:
        if data["Fecha-O"] is None or data["Fecha-I"] is None:
            return None
        fecha_o = datetime.strptime(data["Fecha-O"], "%Y-%m-%d %H:%M:%S")
        fecha_i = datetime.strptime(data["Fecha-I"], "%Y-%m-%d %H:%M:%S")
        min_diff = ((fecha_o - fecha_i).total_seconds()) / 60
        return min_diff

    def fit(self, features: pd.DataFrame, target: pd.DataFrame) -> None:
        """
        Fit model with preprocessed data.

        Args:
            features (pd.DataFrame): preprocessed data.
            target (pd.DataFrame): target.
        """
        x_train, x_val, y_train, y_val = train_test_split(
            features, target, test_size=0.33, random_state=42
        )

        n_y0 = len(y_train[y_train == 0])
        n_y1 = len(y_train[y_train == 1])
        scale = n_y0 / n_y1

        self._model = xgb.XGBClassifier(
            random_state=1, learning_rate=0.01, scale_pos_weight=scale
        )
        self._model.fit(x_train, y_train)
        self._model.save_model(f"{self._save_dir}/model.json")

        y_pred = self._model.predict(x_val)
        print(confusion_matrix(y_val, y_pred))
        print(classification_report(y_val, y_pred))
        return

    def predict(self, features: pd.DataFrame) -> List[int]:
        """
        Predict delays for new flights.

        Args:
            features (pd.DataFrame): preprocessed data.

        Returns:
            (List[int]): predicted targets.
        """
        features = features[self._feature_columns]
        return list(map(int, self._model.predict(features)))
