import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
import xgboost as xgb


class Data:
    def __init__(self, *args):
        attributes = [
            "race", "gender", "age", "time_in_hospital", "num_lab_procedures",
            "num_procedures", "num_medications", "number_outpatient",
            "number_emergency", "number_inpatient", "number_diagnoses",
            "max_glu_serum", "metformin", "repaglinide", "nateglinide",
            "chlorpropamide", "glimepiride", "acetohexamide", "glipizide",
            "glyburide", "tolbutamide", "pioglitazone", "rosiglitazone",
            "acarbose", "miglitol", "troglitazone", "tolazamide", "examide",
            "citoglipton", "insulin", "glyburide_metformin", "glipizide_metformin",
            "glimepiride_pioglitazone", "metformin_rosiglitazone",
            "metformin_pioglitazone", "change", "diabetesMed", "readmitted"
        ]
        for attr, value in zip(attributes, args):
            setattr(self, attr, value)

        data = pd.read_csv("diabetic_data.csv")

        target = data['A1Cresult'].replace({
            'Norm': 5,
            '>7': 7.5,
            '>8': 8.5,
        })
        target = target.fillna(0)

        data = data.drop([
            "encounter_id", "patient_nbr", "weight", "admission_type_id",
            "discharge_disposition_id", "admission_source_id", "payer_code",
            "medical_specialty", "diag_1", "diag_2", "diag_3", "A1Cresult"
        ], axis="columns")

        numerical_columns = data.select_dtypes(include="number").columns
        categorical_columns = data.select_dtypes(include="object").columns

        numerical_pipe = Pipeline(steps=[
            ("mean_impute", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler())
        ])

        categorical_pipe = Pipeline(steps=[
            ("most_frequent_impute", SimpleImputer(strategy="most_frequent")),
            ("one_hot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ])

        self.full_col_transform = ColumnTransformer(transformers=[
            ("num", numerical_pipe, numerical_columns),
            ("cat", categorical_pipe, categorical_columns)
        ])

        data_transform = self.full_col_transform.fit_transform(data)

        x_train_data, _, y_train_data, _ = train_test_split(
            data_transform, target, test_size=0.2, random_state=32
        )

        y_train_data = y_train_data.values.ravel()

        self.model = xgb.XGBRegressor(
            objective='reg:squarederror',
            eval_metric='rmse',
            random_state=42,
            colsample_bytree=0.7,
            gamma=0,
            learning_rate=0.01,
            max_depth=3,
            n_estimators=100,
            subsample=0.7
        )
        self.model.fit(x_train_data, y_train_data)

        self.columns = data.columns

    def predict(self):
        input_dict = {col: [getattr(self, col)] for col in self.columns}
        df = pd.DataFrame(input_dict, columns=self.columns)

        df_transformed = self.full_col_transform.transform(df)

        prediction = self.model.predict(df_transformed)[0]
        return prediction
