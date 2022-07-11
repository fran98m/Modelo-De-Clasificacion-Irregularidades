import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from const import FINES_BASE_PROPERTIES, GENERAL_BASE_PROPERTIES, NO_REQUIRED_COLUMNS, ROOT_DIR
from typing import Any, List

class Base():

    def __init__(self, base_left: pd.DataFrame, base_right: pd.DataFrame, *args, **kargs) -> None:

        self.base_left = base_left
        self.base_right = base_right
        self.exist_temp = False

        if os.path.exists(os.path.join(ROOT_DIR, "files/temp/base.pkl")):
            self.base = pd.read_pickle(os.path.join(ROOT_DIR, "files/temp/base.pkl"))
            self.exist_temp = True
        else:
            self.base = None
            self.__merge_bases()

        self.enconders = {}
        self.scaler = None

    def __getitem__(self, item):
         return self.base[item]

    @property
    def label(self):
        return "irregularidades"

    def __merge_bases(self):
        """join left and right bases
            ...
        """

        #Merge de las bases de datos y multas basado en el numero de contrato
        self.base = pd.merge(self.base_left, 
            self.base_right, 
            on="contrato", 
            how="left")
        
        # Se crea una columna de 0 y 1 teniendo en cuenta si la columna tiene un valor en la sanción o no y luego se la agrega a la base de datos
        #Ademas se reemplazan los missing values en la columna de irregularidades por 0
        self.base[self.label] = (self.base["sancion"] != 0).astype('int32')

        self.base.drop(["contrato", "sancion"], axis=1, inplace=True)
        self.base.columns = [col.lower().replace(' ', '_') for col in self.base.columns]
        self.undersampling(value = 1, 
            column = self.label, 
            reduce = (self.base[self.label].value_counts()[1] - self.base[self.label].value_counts()[0]) / self.base[self.label].value_counts()[1]) #reduce to similar

    def undersampling(self, value: Any, column: str, reduce: float = 0.2):

        mask_value = self[column] == value
        index_to_delete = self.base[mask_value].index.values
        np.random.shuffle(index_to_delete)
        
        self.base.drop(index = index_to_delete[:int(len(index_to_delete) * reduce)], axis = 1, inplace=True)
        self.base.reset_index(drop=True, inplace=True)
    
    def preprocess_text(self, column: str) -> None:
        """Preprocess only the text inside a serie of pandas

        Returns:
            None
        """
        self.base[column].fillna("", inplace=True)
        self.base[column] = self.base[column].str.lower().replace(r"(?!\d)+", "")
    
    def stadarize_numbers(self, columns: List[str], scaler: 'StandardScaler') -> 'pd.Dataframe':
        for col in columns:
            if pd.isna(self[col]).sum() > 0:
                    self.base[col] = SimpleImputer(missing_values=np.nan).fit_transform(self.base[col].values.reshape(-1, 1))

        self.base[columns] = scaler().fit_transform(self.base[columns].values)
        self.scaler = scaler

    def split_data(self) -> 'tuple':

        train_df, val_df, test_df = np.split(self.base.sample(frac=1), 
        [int(.8 * len(self.base)), int(.9 * len(self.base))])

        train_df = pd.DataFrame(train_df, columns=self.base.columns.tolist())
        val_df = pd.DataFrame(val_df, columns=self.base.columns.tolist())
        test_df = pd.DataFrame(test_df, columns=self.base.columns.tolist())

        return train_df, val_df, test_df

    def save(self) -> None:
        if self.base is not None:
            self.base.to_pickle(os.path.join(ROOT_DIR, "files/temp/base.pkl"))

    @classmethod
    def load_data(cls, paths: List[str]) -> 'Base':
        bases = []
        if os.path.exists(os.path.join(ROOT_DIR, "files/temp/base.pkl")):
            return cls(
                base_left = None,
                base_right = None,
            )
        else:
            for path, properties in zip(paths[:2], [FINES_BASE_PROPERTIES, GENERAL_BASE_PROPERTIES]):
                bases.append(pd.read_csv(path, **properties))
            
            return cls(
                base_left = bases[0],
                base_right = bases[1],
            )

    @staticmethod
    def convert_to_preprocess_text(process: 'LabelEncoder|OneHotEncoder', **kargs) ->'function':

        def preprocess(column: pd.Series):
            values = column.values
            if pd.isna(column).sum() > 0:
                values = SimpleImputer(missing_values=np.nan, strategy="most_frequent").fit_transform(values.reshape(-1, 1))

            result = process(**kargs).fit_transform(values)
            return process, result

        return preprocess
    