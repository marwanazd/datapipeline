import logging
import pandas as pd
import mysql.connector
import psycopg2
from sklearn.utils import all_estimators
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder, OrdinalEncoder
from sklearn.base import BaseEstimator, TransformerMixin
import xgboost as xgb
import time
import warnings
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.covariance import EllipticEnvelope
from sklearn.neighbors import LocalOutlierFactor
from sklearn.cluster import DBSCAN
from dateutil import parser
from datetime import timedelta
import os
import difflib
from pandas import DataFrame







# Configure logging

logging.basicConfig(level=logging.INFO,  # Set the minimum level for logging
                    format='%(asctime)s [%(levelname)s] - %(message)s',  # Define log message format
                    filename='pipeline_log.log',  # Specify the log file name
                    filemode='w')  # Set the file mode (w for write)

# Extraction Methods
class ExtractFromLocal:
    def __init__(self, file_path: str = None):
        self.file_path = file_path
        self.logger = logging.getLogger(__name__)

    def extract(self, xml_element: str = None, column_mapping: dict = None):
        if column_mapping is None:
            column_mapping = {None: None}

        if self.file_path is None:
            self.logger.error("File path cannot be None")
            raise ValueError("File path cannot be None")

        file_extension = self.file_path.split('.')[-1].lower()

        try:
            self.logger.info(f"Reading data from file: {self.file_path}")
            if file_extension == 'csv':
                data = pd.read_csv(self.file_path)
            elif file_extension == 'xlsx':
                data = pd.read_excel(self.file_path)
            elif file_extension == 'json':
                data = pd.read_json(self.file_path)
            else:
                self.logger.error(f"Unsupported file format: {file_extension}")
                raise ValueError(f"Unsupported file format: {file_extension}")
            self.logger.info("Data extraction successful")
            return data
        except FileNotFoundError:
            self.logger.error(f"File not found: {self.file_path}")
            raise FileNotFoundError(f"File not found: {self.file_path}")
        except Exception as e:
            self.logger.error(f"Error reading file: {str(e)}")
            raise Exception(f"Error reading file: {str(e)}")


class ExtractFromMysql:
    def __init__(self, username:str=None, password:str=None, host:str=None, database:str=None):
        self.username = username
        self.password = password
        self.host = host
        self.database = database
        self.conn = None  # Initialize the connection as None
        self.logger = logging.getLogger(__name__)  # Create a logger for this class

    def connect(self):
        try:
            self.logger.info("Connecting to MySQL...")
            self.conn = mysql.connector.connect(
                user=self.username,
                password=self.password,
                host=self.host,
                database=self.database
            )
            self.logger.info("Connected to MySQL successfully")
        except Exception as e:
            self.logger.error(f"Error while connecting to MySQL: {str(e)}")

    def extract_table(self, table:str=None):
        if self.conn is None:
            self.logger.error("Connection is not established. Call connect() method first.")
            return None

        try:
            self.logger.info(f"Extracting data from table: {table}...")
            query = f"SELECT * FROM {table}"
            with self.conn.cursor() as cursor:
                cursor.execute(query)
                data = cursor.fetchall()
                column_names = [desc[0] for desc in cursor.description]
                df = pd.DataFrame(data, columns=column_names)
            self.logger.info("Data extraction completed")
            return df
        except Exception as e:
            self.logger.error(f"Error while extracting data: {str(e)}")
            return None

    def close_connection(self):
        if self.conn is not None:
            self.logger.info('Closing database connection...')
            self.conn.close()
            self.logger.info('Database connection closed')


class ExtractFromPostgreSQL:
    def __init__(self, username:str=None, password:str=None, host:str=None, port:int=None, database:str=None):
        self.username = username
        self.password = password
        self.host = host
        self.port = port
        self.database = database
        self.conn = None  # Initialize the connection as None
        self.logger = logging.getLogger(__name__)

    def connect(self):
        try:
            self.logger.info("Connecting to PostgreSQL...")
            # Create a PostgreSQL connection
            self.conn = psycopg2.connect(
                user=self.username,
                password=self.password,
                host=self.host,
                port=self.port,
                database=self.database
            )
            self.logger.info("Connected to PostgreSQL successfully")
            return self
        except Exception as e:
            self.logger.error(f"Error in connect() method: {str(e)}")

    def extract_table(self, table:str=None):
        if self.conn is None:
            self.logger.error("Connection is not established. Call connect() method first.")
            return None

        try:
            self.logger.info(f"Extracting data from table: {table}...")
            query = f"SELECT * FROM {table}"
            with self.conn.cursor() as cursor:
                cursor.execute(query)
                data = cursor.fetchall()
                column_names = [desc[0] for desc in cursor.description]
                df = pd.DataFrame(data, columns=column_names)
            return df
        except Exception as e:
            self.logger.error(f"Error: {str(e)}")
            return None

    def close_connection(self):
        if self.conn is not None:
            self.conn.close()
            self.logger.info('Database connection closed')



# Loading tools

class LoadToLocal:
    def __init__(self, data, method):
        self.data = data
        self.method = method.lower()
        self.logger = logging.getLogger(__name__)

    def load(self, file_path: str = None, file_name=None, sheet_name: str = "Sheet1", xml_element: str = "Data"):
        if not isinstance(self.data, pd.DataFrame):
            self.logger.error("Input data must be a Pandas DataFrame")
            raise ValueError("Input data must be a Pandas DataFrame")

        if file_path is None:
            # Set the default file path to the current working directory
            file_path = os.path.join(os.getcwd(), f"output_file.{self.method}")
            file_path = f"{file_path}\{file_name}"
        else:
            file_path = f"{file_path}\{file_name}"

        if file_path.split('.')[-1].lower() != self.method:
            file_path = f"{file_path}.{self.method}"

        try:
            if self.method == 'csv':
                self.data.to_csv(file_path, index=False)
                self.logger.info(f"Data saved to CSV file: {file_path}")
            elif self.method == 'xlsx':
                writer = pd.ExcelWriter(file_path, engine='xlsxwriter')
                self.data.to_excel(writer, sheet_name=sheet_name, index=False)
                writer.save()
                self.logger.info(f"Data saved to Excel file: {file_path}")
            elif self.method == 'json':
                self.data.to_json(file_path, orient='records', lines=True)
                self.logger.info(f"Data saved to JSON file: {file_path}")
            else:
                raise ValueError("Unsupported file format")
        except Exception as e:
            self.logger.error(f"Error saving data to file: {e}")
            raise ValueError(f"Error saving data to file: {e}")


class LoadToMysql:
    def __init__(self, username, password, host, database):
        self.username = username
        self.password = password
        self.host = host
        self.database = database
        self.conn = None
        self.logger = logging.getLogger(__name__)

    def connect(self):
        try:
            self.logger.info("Connecting to MySQL...")
            self.conn = mysql.connector.connect(
                user=self.username,
                password=self.password,
                host=self.host,
            )
            self.logger.info("Connected to MySQL successfully")
            cursor = self.conn.cursor()

            self.logger.info(f"Checking if database '{self.database}' exists...")
            cursor.execute(f"SHOW DATABASES LIKE '{self.database}'")
            if not cursor.fetchone():
                self.logger.info(f"Database '{self.database}' not found - creating it...")
                cursor.execute(f"CREATE DATABASE {self.database}")
                self.conn.commit()
                self.logger.info(f"Database '{self.database}' created successfully.")
                self.logger.info("Reconnected to MySQL and the created database")
            else:
                self.logger.info(f"Database '{self.database}' exists.")

            cursor.close()
            self.conn.close()
            self.conn = mysql.connector.connect(
                user=self.username,
                password=self.password,
                host=self.host,
                database=self.database
            )
            self.logger.info("MySQL connection is established")
            return self
        except Exception as e:
            self.logger.exception("Error during connection")

    def load_data(self, df, table_name, if_exist='replace'):
        cursor = self.conn.cursor()
        try:
            if if_exist == 'append':
                pass
            elif if_exist == 'replace':
                cursor.execute(f"DROP TABLE IF EXISTS {table_name}")
                self.conn.commit()
            elif if_exist == 'update':
                pass
            else:
                self.logger.error("Cannot load data, check if_exist option")
                raise ValueError("Cannot load data, check if_exist option")

            self.logger.info(f"Checking if table '{table_name}' exists...")
            cursor.execute(f"SHOW TABLES LIKE '{table_name}'")
            if not cursor.fetchone():
                self.logger.info('Table creation started...')
                create_table_query = f"CREATE TABLE IF NOT EXISTS {table_name} ("
                for column in df.columns:
                    if df[column].dtype == 'int64':
                        create_table_query += f"{column} INT,"
                    elif df[column].dtype == 'float64':
                        create_table_query += f"{column} FLOAT,"
                    elif df[column].dtype == 'object':
                        create_table_query += f"{column} VARCHAR(255),"
                    elif df[column].dtype == 'bool':
                        create_table_query += f"{column} TINYINT(1),"
                create_table_query = create_table_query[:-1]  # Remove the trailing comma
                create_table_query += ");"
                cursor.execute(create_table_query)
                self.conn.commit()
                self.logger.info('Table creation successful')

                self.logger.info('Reconnecting to MySQL... ')
                cursor.close()
                self.conn.close()
                self.conn = mysql.connector.connect(
                    user=self.username,
                    password=self.password,
                    host=self.host,
                    database=self.database
                )
                cursor = self.conn.cursor()
                self.logger.info('Reconnected to MySQL and selected database')

            row_values = [tuple(row) for row in df.values]
            column_names = tuple(df.columns)
            formatted_column_names = ', '.join(column_names)

            self.logger.info(f'Loading data to table : {table_name} in database {self.database}')
            for index, row in df.iterrows():
                insert_query = f"INSERT INTO {table_name} ({formatted_column_names}) VALUES {row_values[index]}"
                cursor.execute(insert_query)
            self.conn.commit()
            cursor.close()
            self.conn.close()
            self.logger.info('Data loaded successfully')
        except Exception as e:
            self.logger.exception("Error during data loading")


class LoadToPostgresSql:
    def __init__(self, username, password, host, database, port):
        self.username = username
        self.password = password
        self.host = host
        self.database = database.lower()
        self.port = port
        self.conn = None
        self.logger = logging.getLogger(__name__)

    def connect(self):
        try:
            self.logger.info("Connecting to PostgreSQL...")
            self.conn = psycopg2.connect(
                user=self.username,
                password=self.password,
                host=self.host,
                port=self.port
            )
            self.logger.info("Connected to PostgreSQL successfully")
            cursor = self.conn.cursor()
            self.logger.info(f"Checking if database '{self.database}' exists...")
            cursor.execute("SELECT 1 FROM pg_database WHERE datname = %s;", (self.database,))
            if not cursor.fetchone():
                self.logger.info(f"Database '{self.database}' not found - creating it...")
                cursor.execute("COMMIT")
                cursor.execute(f"CREATE DATABASE {self.database};")
                self.conn.commit()
                self.logger.info(f"Database '{self.database}' created successfully.")
                self.logger.info("Reconnected to PostgreSQL and selected database")
                cursor.close()
                self.conn.close()
            else:
                self.logger.info(f"Database '{self.database}' already exists.")

            self.conn = psycopg2.connect(
                user=self.username,
                password=self.password,
                host=self.host,
                database=self.database,
                port=self.port
            )
            self.logger.info("PostgreSQL connection is established")
            return self
        except Exception as e:
            self.logger.exception("Error during connection")

    def load_data(self, df, table_name, if_exist='replace'):
        cursor = self.conn.cursor()
        try:
            if if_exist == 'append':
                pass
            elif if_exist == 'replace':
                drop_table_query = f"DROP TABLE IF EXISTS {table_name};"
                try:
                    self.logger.info(f"Dropped table {table_name} if it existed.")
                    cursor.execute(drop_table_query)
                    self.conn.commit()
                except Exception as e:
                    self.logger.error(f"Error dropping table: {str(e)}")
                finally:
                    cursor.close()
                    self.logger.info(f"Table '{table_name}' dropped successfully.")
            else:
                self.logger.error("Cannot load data, check if_exist option")
                raise ValueError("Cannot load data, check if_exist option")
            # table creation
            cursor = self.conn.cursor()
            self.logger.info(f"Checking if table '{table_name}' exists...")
            cursor.execute(f"SELECT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = %s);", (table_name,))
            if not cursor.fetchone()[0]:
                self.logger.info(f"Table '{table_name}' does not exist; creating it...")
                create_table_query = f"CREATE TABLE IF NOT EXISTS {table_name} ("
                for column in df.columns:
                    if df[column].dtype == 'int64':
                        create_table_query += f"{column} INT,"
                    elif df[column].dtype == 'float64':
                        create_table_query += f"{column} FLOAT,"
                    elif df[column].dtype == 'object':
                        create_table_query += f"{column} VARCHAR(255),"
                    elif df[column].dtype == 'bool':
                        create_table_query += f"{column} BOOLEAN,"
                create_table_query = create_table_query[:-1]  # Remove the trailing comma
                create_table_query += ");"
                cursor.execute(create_table_query)
                self.conn.commit()
                self.logger.info(f"Table '{table_name}' creation successful")
                cursor.close()
                # Start connecting to table
                self.logger.info('Reconnecting to PostgreSQL...')
                cursor.close()
                self.conn.close()
                self.conn = psycopg2.connect(
                    user=self.username,
                    password=self.password,
                    host=self.host,
                    database=self.database,
                    port=self.port
                )
                cursor = self.conn.cursor()
                self.logger.info('Reconnected to PostgreSQL and selected database')
            else:
                self.logger.info(f"Table '{table_name}' exist")

            row_values = [tuple(row) for row in df.values]
            column_names = tuple(df.columns)
            formatted_column_names = ', '.join(column_names)

            self.logger.info(f'Loading data to table : {table_name} in database {self.database}')
            for index, row in df.iterrows():
                insert_query = f"INSERT INTO {table_name} ({formatted_column_names}) VALUES {row_values[index]}"
                cursor.execute(insert_query)
            self.conn.commit()
            cursor.close()
            self.conn.close()
            self.logger.info('Data loaded successfully')
        except Exception as e:
            self.logger.exception("Error during data loading")



# Transform Methods

class NumericalMissingValuesTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, target_columns, method='drop_rows', constant_value=0):
        self.target_columns = target_columns
        self.method = method
        self.constant_value = constant_value
        self.logger = logging.getLogger(__name__)

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()
        if self.method == 'drop_cols':
            df.drop(self.target_columns, axis=1, inplace=True)
            self.logger.info(f"Dropped columns: {self.target_columns}")
        elif self.method == 'drop_rows':
            df.dropna(subset=self.target_columns, inplace=True)
            self.logger.info(f"Dropped rows with missing values in columns: {self.target_columns}")
        elif self.method == 'most_freq':
            most_freq_values = df[self.target_columns].mode().iloc[0]
            df[self.target_columns] = df[self.target_columns].fillna(most_freq_values)
            self.logger.info(f'Filled missing values in columns {self.target_columns} with most frequent values')
        elif self.method == 'least_freq':
            least_freq_values = df[self.target_columns].apply(lambda col: col.value_counts().idxmin())
            df[self.target_columns] = df[self.target_columns].fillna(least_freq_values)
            self.logger.info(f'Filled missing values in columns {self.target_columns} with least frequent values')
        elif self.method == 'ffill':
            df[self.target_columns] = df[self.target_columns].fillna(method='ffill')
            df[self.target_columns] = df[self.target_columns].fillna(method='bfill')
            df.dropna(subset=self.target_columns, inplace=True)
            self.logger.info(f'Filled missing values in columns {self.target_columns} using forward fill')
        elif self.method == 'bfill':
            df[self.target_columns] = df[self.target_columns].fillna(method='bfill')
            df[self.target_columns] = df[self.target_columns].fillna(method='ffill')
            df.dropna(subset=self.target_columns, inplace=True)
            self.logger.info(f'Filled missing values in columns {self.target_columns} using backward fill')
        elif self.method == 'mean':
            df[self.target_columns] = df[self.target_columns].fillna(df[self.target_columns].mean())
            self.logger.info(f'Filled missing values in columns {self.target_columns} with mean values')
        elif self.method == 'median':
            df[self.target_columns] = df[self.target_columns].fillna(df[self.target_columns].median())
            self.logger.info(f'Filled missing values in columns {self.target_columns} with median values')
        elif self.method == 'linear':
            df[self.target_columns] = df[self.target_columns].interpolate(method='linear')
            self.logger.info(f'Interpolated missing values in columns {self.target_columns} using linear interpolation')
        elif self.method == 'input_value':
            df[self.target_columns] = df[self.target_columns].fillna(self.constant_value)
            self.logger.info(f'Filled missing values in columns {self.target_columns} with constant value: {self.constant_value}')
        return df


class RegressorNumericalMissingValueImputer(BaseEstimator, TransformerMixin):
    def __init__(self, target_column, feature_columns, num_method='mean', target_method=None, cat_method='drop_rows',
                 encoder=None, test_size=0.2, random_state=42, shuffle=True,
                 cols_to_scale=None, scaler=None, pick_model_by='R2', decimal_places=4):
        self.target_column = target_column
        self.feature_columns = feature_columns
        self.num_method = num_method
        self.target_method = target_method
        self.cat_method = cat_method
        self.encoder = encoder
        self.test_size = test_size
        self.random_state = random_state
        self.shuffle = shuffle
        self.cols_to_scale = cols_to_scale
        self.scaler = scaler
        self.pick_model_by = pick_model_by
        self.decimal_places = decimal_places
        self.logger = logging.getLogger(__name__)

    def fit(self, X, y=None):
        self.logger.info('Start RegressorNumericalMissingValueImputer Class')
        # ignore warnings
        warnings.filterwarnings("ignore")
        df = X.copy()

        # Collect important info
        # Separate categorical and numerical columns with missing values
        categorical_columns_with_missing = df.select_dtypes(include=['category', 'object']).columns[
            df.select_dtypes(include=['category', 'object']).isna().any()]
        if self.num_method == 'drop_rows' or self.target_method is not None:
            numerical_columns_with_missing = \
            df.drop(columns=self.target_column).select_dtypes(include=['int64', 'float64']).columns[
                df.drop(columns=self.target_column).select_dtypes(include=['int64', 'float64']).isna().any()]
        else:
            numerical_columns_with_missing = df.select_dtypes(include=['int64', 'float64']).columns[
                df.select_dtypes(include=['int64', 'float64']).isna().any()]
        categorical_columns = X.select_dtypes(include=['category', 'object']).columns
        numerical_columns = X.select_dtypes(include=['int64', 'float64']).columns

        # Step 1: Handling missing values
        self.logger.info('Start Handling missing values')
        categorical_missing_values_transformer = self.CategoricalMissingValuesTransformer(
            target_columns=categorical_columns_with_missing,
            method=self.cat_method
        )

        numerical_missing_values_transformer = self.NumericalMissingValuesTransformer(
            target_columns=numerical_columns_with_missing,
            method=self.num_method
        )

        # if cleaning method for numerical valeus is to drop columns with null values
        if self.num_method == 'drop_cols' or self.target_method is not None:
            if self.target_method == 'drop_cols':
                self.logger.error("You can't handle traget data with drop_cols method")
                raise ValueError("You can't handle traget data with drop_cols method")
            self.logger.info(f"Handling missing values of the target column : {self.target_column}")
            target_numerical_missing_values_transformer = self.NumericalMissingValuesTransformer(
                target_columns=self.target_column,
                method=self.target_method
            )
            steps1 = [
                ('numerical_missing_values', numerical_missing_values_transformer),
                ('categorical_missing_values', categorical_missing_values_transformer),
                ('traget_pre_handling', target_numerical_missing_values_transformer)
            ]
        else:
            steps1 = [
                ('numerical_missing_values', numerical_missing_values_transformer),
                ('categorical_missing_values', categorical_missing_values_transformer)
            ]

        pipeline1 = Pipeline(steps1)
        cleand_data = pipeline1.fit_transform(X)
        self.logger.info("Handling missing values : Done successfully")

        # Step 2: Encoding categorical variables
        self.logger.info("Start Encoding categorical variables")
        categorical_columns = cleand_data.select_dtypes(include=['category', 'object']).columns

        data_coding = self.EncoderTransformer(
            categorical_cols=categorical_columns,
            encoder=self.encoder
        )

        steps2 = [
            ('encode_data', data_coding)
        ]

        pipeline2 = Pipeline(steps2)
        cleand_encoded_data = pipeline2.fit_transform(cleand_data)
        self.logger.info("Encoding categorical variables : Done successfully")

        # Step 3: Features and target Selection
        new_columns = cleand_encoded_data.columns
        self.feature_columns = list(set(self.feature_columns) & set(new_columns))

        features = cleand_encoded_data[self.feature_columns]
        target = cleand_encoded_data[self.target_column]

        # Step 4: Train-Test Split
        self.logger.info("Splitting variables to train and test Sets")
        X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=self.test_size,
                                                            random_state=self.random_state, shuffle=self.shuffle)

        # step 5: Scaling numerical features
        data_scaling = self.ScalerTransformer(
            numerical_cols=self.cols_to_scale,
            scaler=self.scaler
        )

        steps3 = [
            ('scale_data', data_scaling)
        ]

        pipeline3 = Pipeline(steps3)
        if self.cols_to_scale is None or self.scaler is None:
            X_train_scaled, X_test_scaled = X_train, X_test
        else:
            self.logger.info(f"Start Scaling columns : {self.cols_to_scale}")
            X_train_scaled = pipeline3.fit_transform(X_train)
            X_test_scaled = pipeline3.transform(X_test)
            self.logger.info("Column Scaling : Done successfully")

        # Step 6: Models Training
        results = []
        models = []
        removed_regressors = [
            "TheilSenRegressor",
            "ARDRegression",
            "CCA",
            "IsotonicRegression",
            "StackingRegressor",
            "MultiOutputRegressor",
            "MultiTaskElasticNet",
            "MultiTaskElasticNetCV",
            "MultiTaskLasso",
            "MultiTaskLassoCV",
            "PLSCanonical",
            "PLSRegression",
            "RadiusNeighborsRegressor",
            "RegressorChain",
            "VotingRegressor",
        ]
        for name, model_class in all_estimators(type_filter='regressor'):
            if name not in removed_regressors:
                model = model_class()
                models.append((name, model))
        models.append(("XGBRegressor", xgb.XGBRegressor()))
        self.logger.info(f"Start Training regression models")
        for model in models:
            try:
                # create the model
                model_name, model_obj = model

                # training
                start_time = time.time()
                model_obj.fit(X_train_scaled, y_train)
                training_time = time.time() - start_time

                # testing
                start_time = time.time()
                y_pred = model_obj.predict(X_test_scaled)
                validation_time = time.time() - start_time

                # extract results
                mse = mean_squared_error(y_test, y_pred)
                rmse = np.sqrt(mse)
                r2 = r2_score(y_test, y_pred)
                n_samples = X_test.shape[0]  # Number of samples in the test set
                n_predictors = X_test.shape[1]  # Number of features
                adjusted_r2 = 1 - (1 - r2) * (n_samples - 1) / (n_samples - n_predictors - 1)
                mae = mean_absolute_error(y_test, y_pred)
                results.append({
                    'Model': model_obj,
                    'Model_name': model_name,
                    'R2': r2,
                    'Adjusted_R2': adjusted_r2,
                    'MSE': mse,
                    'RMSE': rmse,
                    'MAE': mae,
                    'Training_time': training_time,
                    'Validation_time': validation_time
                })
            except Exception as e:
                self.logger.exception(f"An error occurred for {model_name}")
                continue
        self.logger.info("Models Training : Done successfully")
        # Step 7: Selecting the best model
        self.logger.info("Start Selecting the best model")
        results_df = pd.DataFrame(results)
        # Sort the DataFrame based on the desired characteristic (e.g., RMSE)
        results_df_sorted = results_df.sort_values(by=self.pick_model_by, ascending=False, ignore_index=True)
        # Select the best model (With the biggest desired characteristic)
        self.best_model = results_df_sorted.loc[0, 'Model']
        best_model_name = results_df_sorted.loc[0, 'Model_name']
        self.logger.info(f"The best model selected is : {best_model_name}")
        # Step 8: Re-Train the best model with all data
        self.logger.info("Re-Train the best model with all data...")
        # Scaling numerical features

        if self.cols_to_scale is None or self.scaler is None:
            features_scaled = features
        else:
            features_scaled = pipeline3.fit_transform(features)

        # train model
        self.best_model.fit(features_scaled, target)
        self.logger.info("Model is ready")
        return self

    def transform(self, X):
        X_transformed = X.copy()

        if X[self.target_column].isnull().any():
            self.logger.info("Start setting features of missing values")
            # select missing_X
            missing_rows = X[X[self.target_column].isnull()]
            missing_X = missing_rows[self.feature_columns]

            missing_X_index = missing_X.index

            # Handling missing values in missing_X
            categorical_columns_with_missing = missing_X.select_dtypes(include=['category', 'object']).columns[
                missing_X.select_dtypes(include=['category', 'object']).isna().any()]
            numerical_columns_with_missing = missing_X.select_dtypes(include=['int64', 'float64']).columns[
                missing_X.select_dtypes(include=['int64', 'float64']).isna().any()]
            categorical_columns = missing_X.select_dtypes(include=['category', 'object']).columns

            # Step 1: Handling missing values
            categorical_missing_values_transformer = self.CategoricalMissingValuesTransformer(
                target_columns=categorical_columns_with_missing,
                method=self.cat_method
            )

            numerical_missing_values_transformer = self.NumericalMissingValuesTransformer(
                target_columns=numerical_columns_with_missing,
                method=self.num_method
            )

            # Step 2: Encoding categorical variables
            data_coding = self.EncoderTransformer(
                categorical_cols=categorical_columns,
                encoder=self.encoder
            )

            steps4 = [
                ('numerical_missing_values', numerical_missing_values_transformer),
                ('categorical_missing_values', categorical_missing_values_transformer),
                ('encode_data', data_coding)
            ]

            pipeline4 = Pipeline(steps4)
            X_cleand = pipeline4.fit_transform(X)

            missing_X_index = list(set(missing_X_index) & set(X_cleand.index))

            missing_X = X_cleand.loc[missing_X_index][self.feature_columns]
            self.logger.info("features of missing values Ready")

            # Predict Missing Values of target column
            self.logger.info("Start Predicting Missing Values of target column")
            missing_y_pred = self.best_model.predict(missing_X)
            self.logger.info("Predicting Missing Values of target column : Done successfully")
            # Replace the missing values in the original dataframe with the predicted values
            X_transformed.loc[missing_X_index, self.target_column] = missing_y_pred.round(self.decimal_places)
            self.logger.info(f"Handling Missing values of column :{self.target_column}: Done successfully")

        return X_transformed

    class NumericalMissingValuesTransformer(BaseEstimator, TransformerMixin):
        def __init__(self, target_columns, method='drop_cols', constant_value=0):
            self.target_columns = target_columns
            self.method = method
            self.constant_value = constant_value

        def fit(self, X, y=None):
            return self

        def transform(self, X):
            df = X.copy()
            if self.method == 'drop_cols':
                df.drop(self.target_columns, axis=1, inplace=True)
            elif self.method == 'drop_rows':
                df.dropna(subset=self.target_columns, inplace=True)
            elif self.method == 'most_freq':
                most_freq_values = df[self.target_columns].mode().iloc[0]
                df[self.target_columns] = df[self.target_columns].fillna(most_freq_values)
            elif self.method == 'least_freq':
                least_freq_values = df[self.target_columns].apply(lambda col: col.value_counts().idxmin())
                df[self.target_columns] = df[self.target_columns].fillna(least_freq_values)
            elif self.method == 'ffill':
                df[self.target_columns] = df[self.target_columns].fillna(method='ffill')
                df[self.target_columns] = df[self.target_columns].fillna(method='bfill')
                df.dropna(subset=self.target_columns, inplace=True)
            elif self.method == 'bfill':
                df[self.target_columns] = df[self.target_columns].fillna(method='bfill')
                df[self.target_columns] = df[self.target_columns].fillna(method='ffill')
                df.dropna(subset=self.target_columns, inplace=True)
            elif self.method == 'mean':
                df[self.target_columns] = df[self.target_columns].fillna(df[self.target_columns].mean())
            elif self.method == 'median':
                df[self.target_columns] = df[self.target_columns].fillna(df[self.target_columns].median())
            elif self.method == 'linear':
                df[self.target_columns] = df[self.target_columns].interpolate(method='linear')
            elif self.method == 'input_value':
                df[self.target_columns] = df[self.target_columns].fillna(self.constant_value)
            return df

    class CategoricalMissingValuesTransformer(BaseEstimator, TransformerMixin):
        def __init__(self, target_columns, method, constant_value='Unknown'):
            self.target_columns = target_columns
            self.method = method
            self.constant_value = constant_value

        def fit(self, X, y=None):
            return self

        def transform(self, X):
            df = X.copy()
            if self.method == 'drop_cols':
                df.drop(self.target_columns, axis=1, inplace=True)
            elif self.method == 'drop_rows':
                df.dropna(subset=self.target_columns, inplace=True)
            elif self.method == 'most_freq':
                most_freq_values = df[self.target_columns].mode().iloc[0]
                df[self.target_columns] = df[self.target_columns].fillna(most_freq_values)
            elif self.method == 'least_freq':
                least_freq_values = df[self.target_columns].apply(lambda col: col.value_counts().idxmin())
                df[self.target_columns] = df[self.target_columns].fillna(least_freq_values)
            elif self.method == 'ffill':
                df[self.target_columns] = df[self.target_columns].fillna(method='ffill')
                df[self.target_columns] = df[self.target_columns].fillna(method='bfill')
                df.dropna(subset=self.target_columns, inplace=True)
            elif self.method == 'bfill':
                df[self.target_columns] = df[self.target_columns].fillna(method='bfill')
                df[self.target_columns] = df[self.target_columns].fillna(method='ffill')
                df.dropna(subset=self.target_columns, inplace=True)
            elif self.method == 'input_value':
                df[self.target_columns] = df[self.target_columns].fillna(self.constant_value)
            return df

    class ScalerTransformer(BaseEstimator, TransformerMixin):
        def __init__(self, numerical_cols, scaler):
            if isinstance(numerical_cols, str):
                self.numerical_cols = [numerical_cols]
            else:
                self.numerical_cols = numerical_cols
            self.scaler = scaler

        def fit(self, X, y=None):
            self.scaler.fit(X[self.numerical_cols])
            return self

        def transform(self, X):
            X_transformed = X.copy()
            # Apply your custom function to the data
            X_transformed[self.numerical_cols] = self.scaler.transform(X_transformed[self.numerical_cols])
            return X_transformed

    class EncoderTransformer(BaseEstimator, TransformerMixin):
        def __init__(self, categorical_cols, encoder):
            if isinstance(categorical_cols, str):
                self.categorical_cols = [categorical_cols]
            else:
                self.categorical_cols = categorical_cols
            self.encoder = encoder

        def fit(self, X, y=None):
            return self

        def transform(self, X):
            X_transformed = X.copy()
            for col in self.categorical_cols:
                X_transformed[col] = self.encoder.fit_transform(X_transformed[col])
            return X_transformed


class CategoricalMissingValuesTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, target_columns, method='drop_rows', constant_value='Unknown'):
        self.target_columns = target_columns
        self.method = method
        self.constant_value = constant_value
        self.logger = logging.getLogger(__name__)

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()
        if self.method == 'drop_cols':
            self.logger.info(f"Before dropping columns: {df.shape}")
            df.drop(self.target_columns, axis=1, inplace=True)
            self.logger.info(f"Dropped columns: {self.target_columns}")
            self.logger.info(f"After dropping columns: {df.shape}")
        elif self.method == 'drop_rows':
            self.logger.info(f"Before dropping rows with missing values: {df.shape}")
            df.dropna(subset=self.target_columns, inplace=True)
            self.logger.info(f"Dropped rows with missing values in columns: {self.target_columns}")
            self.logger.info(f"After dropping rows with missing values: {df.shape}")
        elif self.method == 'most_freq':
            most_freq_values = df[self.target_columns].mode().iloc[0]
            self.logger.info(f"Before filling missing values with most frequent values: {df.shape}")
            df[self.target_columns] = df[self.target_columns].fillna(most_freq_values)
            self.logger.info(f'Filled missing values in columns {self.target_columns} with most frequent values')
            self.logger.info(f"After filling missing values with most frequent values: {df.shape}")
        elif self.method == 'least_freq':
            least_freq_values = df[self.target_columns].apply(lambda col: col.value_counts().idxmin())
            self.logger.info(f"Before filling missing values with least frequent values: {df.shape}")
            df[self.target_columns] = df[self.target_columns].fillna(least_freq_values)
            self.logger.info(f'Filled missing values in columns {self.target_columns} with least frequent values')
            self.logger.info(f"After filling missing values with least frequent values: {df.shape}")
        elif self.method == 'ffill':
            self.logger.info(f"Before filling missing values with forward fill: {df.shape}")
            df[self.target_columns] = df[self.target_columns].fillna(method='ffill')
            df[self.target_columns] = df[self.target_columns].fillna(method='bfill')
            df.dropna(subset=self.target_columns, inplace=True)
            self.logger.info(f'Filled missing values in columns {self.target_columns} using forward fill and backward fill')
            self.logger.info(f"After filling missing values with forward fill and backward fill: {df.shape}")
        elif self.method == 'bfill':
            self.logger.info(f"Before filling missing values with backward fill: {df.shape}")
            df[self.target_columns] = df[self.target_columns].fillna(method='bfill')
            df[self.target_columns] = df[self.target_columns].fillna(method='ffill')
            df.dropna(subset=self.target_columns, inplace=True)
            self.logger.info(f'Filled missing values in columns {self.target_columns} using backward fill and forward fill')
            self.logger.info(f"After filling missing values with backward fill and forward fill: {df.shape}")
        elif self.method == 'input_value':
            self.logger.info(f"Before filling missing values with constant value: {df.shape}")
            df[self.target_columns] = df[self.target_columns].fillna(self.constant_value)
            self.logger.info(f'Filled missing values in columns {self.target_columns} with constant value: {self.constant_value}')
            self.logger.info(f"After filling missing values with constant value: {df.shape}")
        return df


class ClassifierCategoricalMissingValueImputer(BaseEstimator, TransformerMixin):
    def __init__(self, target_column, feature_columns, num_method='mean', cat_method='drop_rows',
                 target_method='bfill', encoder=None, test_size=0.2, random_state=42, shuffle=True,
                 cols_to_scale=None, scaler=None, pick_model_by='accuracy'):
        self.target_column = target_column
        self.feature_columns = feature_columns
        self.num_method = num_method
        self.cat_method = cat_method
        self.target_method = target_method
        self.encoder = encoder
        self.test_size = test_size
        self.random_state = random_state
        self.shuffle = shuffle
        self.cols_to_scale = cols_to_scale
        self.scaler = scaler
        self.pick_model_by = pick_model_by
        self.logger = logging.getLogger(__name__)

    def fit(self, X, y=None):
        self.logger.info('Start RegressorNumericalMissingValueImputer Class')
        # ignore warnings
        warnings.filterwarnings("ignore")

        df = X.copy()

        # Collect important info
        # Separate categorical and numerical columns with missing values
        categorical_columns_with_missing = df.select_dtypes(include=['category', 'object']).columns[
            df.select_dtypes(include=['category', 'object']).isna().any()]
        if self.num_method == 'drop_rows' or self.target_method is not None:
            numerical_columns_with_missing = \
            df.drop(columns=self.target_column).select_dtypes(include=['int64', 'float64']).columns[
                df.drop(columns=self.target_column).select_dtypes(include=['int64', 'float64']).isna().any()]
        else:
            numerical_columns_with_missing = df.select_dtypes(include=['int64', 'float64']).columns[
                df.select_dtypes(include=['int64', 'float64']).isna().any()]
        categorical_columns = X.select_dtypes(include=['category', 'object']).columns
        numerical_columns = X.select_dtypes(include=['int64', 'float64']).columns

        # Step 1: Handling missing values
        self.logger.info('Start Handling missing values')
        categorical_missing_values_transformer = self.CategoricalMissingValuesTransformer(
            target_columns=categorical_columns_with_missing,
            method=self.cat_method
        )

        numerical_missing_values_transformer = self.NumericalMissingValuesTransformer(
            target_columns=numerical_columns_with_missing,
            method=self.num_method
        )

        # if cleaning method for numerical valeus is to drop columns with null values
        if self.num_method == 'drop_cols' or self.target_method is not None:
            if self.target_method == 'drop_cols':
                self.logger.error("You can't handle traget data with drop_cols method")
                raise ValueError("You can't handle traget data with drop_cols method")
            self.logger.info(f"Handling missing values of the target column : {self.target_column}")
            target_numerical_missing_values_transformer = self.NumericalMissingValuesTransformer(
                target_columns=self.target_column,
                method=self.target_method
            )
            steps1 = [
                ('numerical_missing_values', numerical_missing_values_transformer),
                ('categorical_missing_values', categorical_missing_values_transformer),
                ('traget_pre_handling', target_numerical_missing_values_transformer)
            ]
        else:
            steps1 = [
                ('numerical_missing_values', numerical_missing_values_transformer),
                ('categorical_missing_values', categorical_missing_values_transformer)
            ]

        pipeline1 = Pipeline(steps1)
        cleand_data = pipeline1.fit_transform(X)
        self.logger.info("Handling missing values : Done successfully")

        # Step 2: Encoding categorical variables
        self.logger.info("Start Encoding categorical variables")
        categorical_columns = cleand_data.drop(columns=self.target_column).select_dtypes(
            include=['category', 'object']).columns

        self.data_coding = self.EncoderTransformer(
            categorical_cols=categorical_columns,
            encoder=self.encoder
        )

        steps2 = [
            ('encode_data', self.data_coding)
        ]

        pipeline2 = Pipeline(steps2)
        cleand_encoded_data = pipeline2.fit_transform(cleand_data)
        self.logger.info("Encoding categorical variables : Done successfully")
        # Step 3: Features and target Selection
        new_columns = cleand_encoded_data.columns
        self.feature_columns = list(set(self.feature_columns) & set(new_columns))

        features = cleand_encoded_data[self.feature_columns]
        target = cleand_encoded_data[self.target_column]

        # Step 4: Train-Test Split
        self.logger.info("Splitting variables to train and test Sets")
        X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=self.test_size,
                                                            random_state=self.random_state, shuffle=self.shuffle)

        # step 5: Scaling numerical features
        data_scaling = self.ScalerTransformer(
            numerical_cols=self.cols_to_scale,
            scaler=self.scaler
        )

        steps3 = [
            ('scale_data', data_scaling)
        ]

        pipeline3 = Pipeline(steps3)
        if self.cols_to_scale is None or self.scaler is None:
            X_train_scaled, X_test_scaled = X_train, X_test
        else:
            self.logger.info(f"Start Scaling columns : {self.cols_to_scale}")
            X_train_scaled = pipeline3.fit_transform(X_train)
            X_test_scaled = pipeline3.transform(X_test)
            self.logger.info("Column Scaling : Done successfully")

        # Step 6: Models Training
        results = []
        models = []
        removed_classifiers = [
            "ClassifierChain",
            "ComplementNB",
            "GradientBoostingClassifier",
            "GaussianProcessClassifier",
            "HistGradientBoostingClassifier",
            "MLPClassifier",
            "LogisticRegressionCV",
            "MultiOutputClassifier",
            "MultinomialNB",
            "OneVsOneClassifier",
            "OneVsRestClassifier",
            "OutputCodeClassifier",
            "RadiusNeighborsClassifier",
            "VotingClassifier",
            "StackingClassifier",
        ]

        for name, model_class in all_estimators(type_filter='classifier'):
            if name not in removed_classifiers:
                model = model_class()
                models.append((name, model))
        models.append(("XGBClassifier", xgb.XGBClassifier()))
        self.logger.info(f"Start Training regression models")
        for model in models:
            try:
                # create the model
                model_name, model_obj = model
                if model_name == 'XGBClassifier':

                    labels = list(set(y_train).union(set(y_test)))
                    self.label_encoder = LabelEncoder()
                    self.label_encoder.fit(labels)
                    y_train_xgb = self.label_encoder.transform(y_train)
                    # training
                    start_time = time.time()
                    model_obj.fit(X_train_scaled, y_train_xgb)
                    training_time = time.time() - start_time
                else:
                    # training
                    start_time = time.time()
                    model_obj.fit(X_train_scaled, y_train)
                    training_time = time.time() - start_time

                # testing
                start_time = time.time()
                y_pred = model_obj.predict(X_test_scaled)
                validation_time = time.time() - start_time

                if model_name == 'XGBClassifier':
                    y_pred = self.label_encoder.inverse_transform(y_pred)
                # extract results
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, average='weighted')
                recall = recall_score(y_test, y_pred, average='weighted')
                f1 = f1_score(y_test, y_pred, average='weighted')
                results.append({
                    'Model': model_obj,
                    'Model_name': model_name,
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1': f1,
                    'Training_time': training_time,
                    'Validation_time': validation_time
                })
            except Exception as e:
                self.logger.exception(f"An error occurred for {model_name}")
                continue

        self.logger.info("Models Training : Done successfully")
        # Step 7: Selecting the best model
        self.logger.info("Start Selecting the best model")
        results_df = pd.DataFrame(results)
        # Sort the DataFrame based on the desired characteristic (e.g., RMSE)
        results_df_sorted = results_df.sort_values(by=self.pick_model_by, ascending=False, ignore_index=True)
        # Select the best model (With the biggest desired characteristic)
        self.best_model = results_df_sorted.loc[0, 'Model']
        self.name_best_model = results_df_sorted.loc[0, 'Model_name']
        best_model_name = results_df_sorted.loc[0, 'Model_name']
        self.logger.info(f"The best model selected is : {best_model_name}")
        # Step 8: Re-Train the best model with all data
        self.logger.info("Re-Train the best model with all data...")
        # Scaling numerical features

        if self.cols_to_scale is None or self.scaler is None:
            features_scaled = features
        else:
            features_scaled = pipeline3.fit_transform(features)

        if self.name_best_model == 'XGBClassifier':
            target_xgb = self.label_encoder.transform(target)
            # train model
            self.best_model.fit(features_scaled, target_xgb)
        else:
            # train model
            self.best_model.fit(features_scaled, target)
        self.logger.info("Model is ready")
        return self

    def transform(self, X):
        X_transformed = X.copy()

        if X[self.target_column].isnull().any():
            self.logger.info("Start setting features of missing values")
            # select missing_X
            missing_rows = X[X[self.target_column].isnull()]
            missing_X = missing_rows[self.feature_columns]

            missing_X_index = missing_X.index

            # Handling missing values in missing_X
            categorical_columns_with_missing = missing_X.select_dtypes(include=['category', 'object']).columns[
                missing_X.select_dtypes(include=['category', 'object']).isna().any()]
            numerical_columns_with_missing = missing_X.select_dtypes(include=['int64', 'float64']).columns[
                missing_X.select_dtypes(include=['int64', 'float64']).isna().any()]
            categorical_columns = missing_X.select_dtypes(include=['category', 'object']).columns

            # Step 1: Handling missing values
            categorical_missing_values_transformer = self.CategoricalMissingValuesTransformer(
                target_columns=categorical_columns_with_missing,
                method=self.cat_method
            )

            numerical_missing_values_transformer = self.NumericalMissingValuesTransformer(
                target_columns=numerical_columns_with_missing,
                method=self.num_method
            )

            # Step 2: Encoding categorical variables
            data_coding = self.EncoderTransformer(
                categorical_cols=categorical_columns,
                encoder=self.encoder
            )

            steps4 = [
                ('numerical_missing_values', numerical_missing_values_transformer),
                ('categorical_missing_values', categorical_missing_values_transformer),
                ('encode_data', data_coding)
            ]

            pipeline4 = Pipeline(steps4)
            X_cleand = pipeline4.fit_transform(X)

            missing_X_index = list(set(missing_X_index) & set(X_cleand.index))

            missing_X = X_cleand.loc[missing_X_index][self.feature_columns]
            self.logger.info("features of missing values Ready")
            # Predict Missing Values of target column
            self.logger.info("Start Predicting Missing Values of target column")
            missing_y_pred = self.best_model.predict(missing_X)
            self.logger.info("Predicting Missing Values of target column : Done successfully")

            # decode missing_y_pred
            if self.name_best_model == 'XGBClassifier':
                missing_y_pred = self.label_encoder.inverse_transform(missing_y_pred)

            # Replace the missing values in the original dataframe with the predicted values
            X_transformed.loc[missing_X_index, self.target_column] = missing_y_pred
            self.logger.info(f"Handling Missing values of column :{self.target_column}: Done successfully")

        return X_transformed

    class NumericalMissingValuesTransformer(BaseEstimator, TransformerMixin):
        def __init__(self, target_columns, method='drop_rows', constant_value=0):
            self.target_columns = target_columns
            self.method = method
            self.constant_value = constant_value

        def fit(self, X, y=None):
            return self

        def transform(self, X):
            df = X.copy()
            if self.method == 'drop_cols':
                df.drop(self.target_columns, axis=1, inplace=True)
            elif self.method == 'drop_rows':
                df.dropna(subset=self.target_columns, inplace=True)
            elif self.method == 'most_freq':
                most_freq_values = df[self.target_columns].mode().iloc[0]
                df[self.target_columns] = df[self.target_columns].fillna(most_freq_values)
            elif self.method == 'least_freq':
                least_freq_values = df[self.target_columns].apply(lambda col: col.value_counts().idxmin())
                df[self.target_columns] = df[self.target_columns].fillna(least_freq_values)
            elif self.method == 'ffill':
                df[self.target_columns] = df[self.target_columns].fillna(method='ffill')
                df[self.target_columns] = df[self.target_columns].fillna(method='bfill')
                df.dropna(subset=self.target_columns, inplace=True)
            elif self.method == 'bfill':
                df[self.target_columns] = df[self.target_columns].fillna(method='bfill')
                df[self.target_columns] = df[self.target_columns].fillna(method='ffill')
                df.dropna(subset=self.target_columns, inplace=True)
            elif self.method == 'mean':
                df[self.target_columns] = df[self.target_columns].fillna(df[self.target_columns].mean())
            elif self.method == 'median':
                df[self.target_columns] = df[self.target_columns].fillna(df[self.target_columns].median())
            elif self.method == 'linear':
                df[self.target_columns] = df[self.target_columns].interpolate(method='linear')
            elif self.method == 'input_value':
                df[self.target_columns] = df[self.target_columns].fillna(self.constant_value)
            return df

    class CategoricalMissingValuesTransformer(BaseEstimator, TransformerMixin):
        def __init__(self, target_columns, method='drop_rows', constant_value='Unknown'):
            self.target_columns = target_columns
            self.method = method
            self.constant_value = constant_value

        def fit(self, X, y=None):
            return self

        def transform(self, X):
            df = X.copy()
            if self.method == 'drop_cols':
                df.drop(self.target_columns, axis=1, inplace=True)
            elif self.method == 'drop_rows':
                df.dropna(subset=self.target_columns, inplace=True)
            elif self.method == 'most_freq':
                most_freq_values = df[self.target_columns].mode().iloc[0]
                df[self.target_columns] = df[self.target_columns].fillna(most_freq_values)
            elif self.method == 'least_freq':
                least_freq_values = df[self.target_columns].apply(lambda col: col.value_counts().idxmin())
                df[self.target_columns] = df[self.target_columns].fillna(least_freq_values)
            elif self.method == 'ffill':
                df[self.target_columns] = df[self.target_columns].fillna(method='ffill')
                df.dropna(subset=self.target_columns, inplace=True)
            elif self.method == 'bfill':
                df[self.target_columns] = df[self.target_columns].fillna(method='bfill')
                df.dropna(subset=self.target_columns, inplace=True)
            elif self.method == 'input_value':
                df[self.target_columns] = df[self.target_columns].fillna(self.constant_value)
            return df

    class ScalerTransformer(BaseEstimator, TransformerMixin):
        def __init__(self, numerical_cols, scaler):
            if isinstance(numerical_cols, str):
                self.numerical_cols = [numerical_cols]
            else:
                self.numerical_cols = numerical_cols
            self.scaler = scaler

        def fit(self, X, y=None):
            self.scaler.fit(X[self.numerical_cols])
            return self

        def transform(self, X):
            X_transformed = X.copy()
            # Apply your custom function to the data
            X_transformed[self.numerical_cols] = self.scaler.transform(X_transformed[self.numerical_cols])
            return X_transformed

    class EncoderTransformer(BaseEstimator, TransformerMixin):
        def __init__(self, categorical_cols, encoder):
            if isinstance(categorical_cols, str):
                self.categorical_cols = [categorical_cols]
            else:
                self.categorical_cols = categorical_cols
            self.encoder = encoder
            self.inverse_encoders = {}

        def fit(self, X, y=None):
            return self

        def transform(self, X):
            X_transformed = X.copy()
            for col in self.categorical_cols:
                X_transformed[col] = self.encoder.fit_transform(X_transformed[col])
            return X_transformed


class DropDuplicatesTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, columns: bool = True, rows: bool = True):
        self.columns = columns
        self.rows = rows
        self.logger = logging.getLogger(__name__)

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()
        self.logger.info("Transforming data with DropDuplicatesTransformer...")
        columns_to_keep = []
        columns_to_drop = []
        duplicated_cols = set()  # Initialize a set to keep track of duplicated columns

        # Iterate over all pairs of columns to check for duplicated columns
        for i, col1 in enumerate(df.columns):
            for j, col2 in enumerate(df.columns):
                # Compare the two columns using equals()
                if col1 in df.columns:
                    if col2 not in columns_to_keep and col1 not in columns_to_drop and i != j and df[col1].equals(
                            df[col2]):
                        columns_to_keep.append(col1)
                        columns_to_drop.append(col2)

        # Drop duplicate rows
        if df.duplicated().any() and self.rows:
            self.logger.info("Dropping duplicate rows...")
            df.drop_duplicates(inplace=True)
            df.reset_index(drop=True, inplace=True)

        # Drop duplicate columns
        if len(columns_to_drop) > 0 and self.columns:
            self.logger.info("Dropping duplicate columns...")
            df = df.drop(columns=columns_to_drop)

        self.logger.info("Transformation completed.")
        return df

class FilteringDataTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, by='index', condition=(0, 0)):
        self.by = by
        self.condition = condition
        self.logger = logging.getLogger(__name__)

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        self.logger.info("Transforming data with FilteringDataTransformer...")
        df = X.copy()
        if self.condition[0] < self.condition[1]:
            if self.by == 'index':
                self.logger.info(f"Filtering rows by index: {self.condition[0]} - {self.condition[1]}")
                df = df.iloc[self.condition[0]:self.condition[1] + 1, :]
            elif self.by in df.columns:
                self.logger.info(f"Filtering rows by {self.by} between {self.condition[0]} and {self.condition[1]}")
                mask = df[self.by].between(self.condition[0], self.condition[1])
                df = df.loc[mask]
        elif self.condition[0] == self.condition[1]:
            if self.by == 'index':
                self.logger.info(f"Selecting row at index: {self.condition[0]}")
                df = df.iloc[self.condition[0]:self.condition[1] + 1, :]
            elif self.by in df.columns:
                self.logger.info(f"Selecting rows where {self.by} is equal to {self.condition[0]}")
                mask = df[self.by] == self.condition[0]
                df = df.loc[mask]

        self.logger.info("Transformation completed.")
        return df


class ScalerTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, numerical_cols, scaler):
        if isinstance(numerical_cols, str):
            self.numerical_cols = [numerical_cols]
        else:
            self.numerical_cols = numerical_cols
        self.scaler = scaler
        self.logger = logging.getLogger(__name__)

    def fit(self, X, y=None):
        self.logger.info(f"Fitting scaler :{self.scaler}:...")
        self.scaler.fit(X[self.numerical_cols])
        self.logger.info("Scaler fitting completed.")
        return self

    def transform(self, X):
        self.logger.info("Transforming data with ScalerTransformer...")
        X_transformed = X.copy()
        # Apply your custom function to the data
        self.logger.info(f"Scaling data with scaler :{self.scaler}:...")
        X_transformed[self.numerical_cols] = self.scaler.transform(X_transformed[self.numerical_cols])
        self.logger.info("Scaling completed.")
        return X_transformed


class EncoderTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, categorical_cols, encoder):
        if isinstance(categorical_cols, str):
            self.categorical_cols = [categorical_cols]
        else:
            self.categorical_cols = categorical_cols
        self.encoder = encoder
        self.logger = logging.getLogger(__name__)

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        self.logger.info("Transforming data with EncoderTransformer...")
        X_transformed = X.copy()
        for col in self.categorical_cols:
            self.logger.info(f"Fitting encoder :{self.encoder}: for column: {col}...")
            X_transformed[col] = self.encoder.fit_transform(X_transformed[col])
            self.logger.info(f"Encoding completed.")
        return X_transformed


class ColumnRenamingTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, column_name_mapping:dict, auto_finder:bool=False):
        self.column_name_mapping = column_name_mapping
        self.auto_finder = auto_finder
        self.logger = logging.getLogger(__name__)

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        self.logger.info("Transforming data with ColumnRenamingTransformer...")
        df = X.copy()
        if self.auto_finder is True:
            self.logger.info("Start automatic column match finder...")
            check_results = {}
            for key in self.column_name_mapping.keys():
                match = difflib.get_close_matches(key, df.columns, n=1, cutoff=0.1)
                check_results[key] = match[0]
            self.logger.info("Automatic column match finder : Done")
            self.column_name_mapping = check_results
        self.logger.info("Transforming data with ColumnRenamingTransformer...")
        df = X.copy()
        self.logger.info(f"Column renaming start...")
        df = df.rename(columns=self.column_name_mapping)
        self.logger.info("Column renaming completed.")
        return df


class DataPivotingTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, column_name, drop_categories=True):
        self.column_name = column_name
        self.drop_categories = drop_categories
        self.logger = logging.getLogger(__name__)

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        self.logger.info(f"Transforming data with pivoting on column '{self.column_name}'...")
        df = X.copy()
        split_dataframes = []

        unique_values = df[self.column_name].unique()
        for value in unique_values:
            split_df = df[df[self.column_name] == value].reset_index(drop=True)
            split_df.columns = [f"{col}_{value}" for col in split_df.columns]
            split_dataframes.append(split_df)

        grouped_df = pd.concat(split_dataframes, axis=1)
        if self.drop_categories:
            # Initialize an empty list to store columns to drop
            columns_to_drop = []
            # Iterate through columns in the DataFrame
            for column in grouped_df.columns:
                # Check if all values in the column are the same
                if grouped_df[column].nunique() == 1:
                    columns_to_drop.append(column)
            # Drop the columns with the same repeated value
            grouped_df = grouped_df.drop(columns=columns_to_drop)
        self.logger.info("Data transformation complete.")
        return grouped_df


class DateTimeFeatureExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, date_column, format_code='auto', extract_features=None, drop=False, columns_position='right'):
        self.date_column = date_column
        self.format_code = format_code
        self.extract_features = extract_features
        self.drop = drop
        self.columns_position = columns_position
        self.logger = logging.getLogger(__name__)

    def fit(self, X, y=None):
        self.logger.info("Fitting DateTimeFeatureExtractor...")
        return self

    @staticmethod
    def time_handler(date_string):
        pm = False
        # Check if the string contains "AM" or "PM"
        if "AM" in date_string:
            index_of_am = date_string.find("AM")
            if index_of_am != -1 and index_of_am > 0:
                char_before_am = date_string[index_of_am - 1]

                if char_before_am.isdigit() or char_before_am.isalpha():
                    date_string = date_string.replace("AM", '')
                else:
                    date_string = date_string.replace(f"{char_before_am}AM", '')

        if "PM" in date_string:
            index_of_pm = date_string.find("PM")
            if index_of_pm != -1 and index_of_pm > 0:
                char_before_pm = date_string[index_of_pm - 1]

                if char_before_pm.isdigit() or char_before_pm.isalpha():
                    date_string = date_string.replace("PM", '')
                else:
                    date_string = date_string.replace(f"{char_before_pm}PM", '')
                pm = True

        # Now parse the date string
        parsed_date = parser.parse(date_string)
        # Check if it's PM and not midnight (12:00 AM)
        if pm and parsed_date.hour != 12:
            parsed_date = parsed_date + timedelta(hours=12)
        return parsed_date

    def transform(self, X):
        self.logger.info("Transforming data with DateTimeFeatureExtractor...")
        df = X.copy()
        parsing = False
        if self.format_code == 'auto':
            try:
                self.logger.info("Automatically parsing date/time strings...")
                df[self.date_column] = df[self.date_column].apply(self.time_handler)
                parsing = True
            except ValueError as e:
                self.logger.error(f"Error parsing date string. error {e}")
        else:
            try:
                self.logger.info(f"Parsing date strings using format code : {self.format_code}...")
                df[self.date_column] = pd.to_datetime(df[self.date_column], format=self.format_code)
                parsing = True
            except ValueError as e:
                self.logger.error(f"Error parsing date string. error {e}")

        if self.extract_features is not None and parsing == True:
            for feature in self.extract_features:
                if feature == 'Year':
                    df['Year'] = df[self.date_column].dt.year
                    self.logger.info("Extracting feature: Year")
                elif feature == 'Month':
                    df['Month'] = df[self.date_column].dt.month
                    self.logger.info("Extracting feature: Month")
                # Add more features here as needed

            if self.columns_position == 'left':
                df = df[self.extract_features + df.columns.difference(self.extract_features).tolist()]
                self.logger.info("Rearranging columns to move selected features to the left.")

            if self.drop == True:
                df.drop(columns=self.date_column, inplace=True)
                self.logger.info(f"Dropped column: ({self.date_column}) Successfully.")

        self.logger.info("Transformation completed.")
        return df


class OutlierHandlingTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, column_name, outlier_indicator='Z-Score', method='drop', drop_indicator=True, fill_value=None):
        self.column_name = column_name
        self.outlier_indicator = outlier_indicator
        self.method = method
        self.drop_indicator = drop_indicator
        self.fill_value = fill_value
        self.logger = logging.getLogger(__name__)

    def fit(self, X, y=None):
        self.logger.info("Fitting OutlierHandlingTransformer...")
        return self

    def transform(self, X):
        self.logger.info("Transforming data with OutlierHandlingTransformer...")
        df = X.copy()

        # Using IQR to Estimate Outliers
        q1 = df[self.column_name].quantile(0.25)
        q3 = df[self.column_name].quantile(0.75)
        iqr = q3 - q1
        minimum = q1 - 1.5 * iqr
        maximum = q3 + 1.5 * iqr
        outlier_predictions = np.where(df[[self.column_name]] < minimum, -1,
                                       np.where(df[[self.column_name]] > maximum, -1, 1))
        df['Interquartile range'] = outlier_predictions
        self.logger.info("Outliers detected using Interquartile range.")

        # Using Isolation Forest Algorithm to Estimate Outliers
        outlier_predictions = IsolationForest().fit(df[[self.column_name]]).predict(df[[self.column_name]])
        df['Isolation Forest'] = outlier_predictions
        self.logger.info("Outliers detected using Isolation Forest Algorithm.")

        # Using Elliptic Envelope Algorithm to Estimate Outliers
        outlier_predictions = EllipticEnvelope().fit(df[[self.column_name]]).predict(df[[self.column_name]])
        df['Elliptic Envelope'] = outlier_predictions
        self.logger.info("Outliers detected using Elliptic Envelope Algorithm.")

        # Using Local Outlier Factor Algorithm to Estimate Outliers
        outlier_predictions = LocalOutlierFactor(n_neighbors=5, novelty=True).fit(df[[self.column_name]]).predict(
            df[[self.column_name]])
        df['Local Outlier'] = outlier_predictions
        self.logger.info("Outliers detected using Local Outlier Factor Algorithm.")

        # DBSCAN (Density-Based Spatial Clustering of Applications with Noise)
        outlier_predictions = DBSCAN(eps=900, min_samples=5).fit_predict(df[[self.column_name]])
        for i, j in enumerate(outlier_predictions):
            if j == 0:
                outlier_predictions[i] = 1
        df['DBSCAN'] = outlier_predictions
        self.logger.info("Outliers detected using DBSCAN Algorithm.")

        # Z-Score
        z_scores = (df[[self.column_name]] - df[[self.column_name]].mean()) / df[[self.column_name]].std()
        threshold = 1.5
        outlier_predictions = np.where(z_scores.abs() > threshold, -1, 1)
        df['Z-Score'] = outlier_predictions
        self.logger.info("Outliers detected using Z-Score.")

        # The Modified Z-Score
        median = df[[self.column_name]].median()
        mad = np.median(np.abs(df[[self.column_name]] - median))
        modified_z_scores = 0.6745 * (df[[self.column_name]] - median) / mad
        threshold = 3.5
        outlier_predictions = np.where(modified_z_scores.abs() > threshold, -1, 1)
        df['The Modified Z-Score'] = outlier_predictions
        self.logger.info("Outliers detected using The Modified Z-Score.")

        # Change -1 to 1 and 1 to 0 for all columns indicating outliers
        outlier_columns = ['Interquartile range', 'Isolation Forest', 'Elliptic Envelope', 'Local Outlier', 'DBSCAN',
                           'Z-Score', 'The Modified Z-Score']
        for col in outlier_columns:
            df[col] = df[col].map({-1: 1, 1: 0})

        # Get the indices of outliers
        outlier_indices = df[df[self.outlier_indicator] == 1].index

        if self.method == 'mean':
            # Impute outliers with the mean of the column
            df.loc[outlier_indices, self.column_name] = df[self.column_name].mean().round(3)
            self.logger.info("Imputed outliers with the mean of the column.")

        elif self.method == 'median':
            # Or replace outliers with the median of the column
            df.loc[outlier_indices, self.column_name] = df[self.column_name].median().round(3)
            self.logger.info("Replaced outliers with the median of the column.")

        elif self.method == 'drop':
            # Or remove rows with outliers
            df = df[~df.index.isin(outlier_indices)]
            self.logger.info("Dropped rows with outliers.")

        elif self.method == 'input':
            # Fill with a specific value
            df.loc[outlier_indices, self.column_name] = self.fill_value
            self.logger.info(f"Filled outliers with the specified value: {self.fill_value}")

        else:
            self.logger.warning("Invalid outlier handling method. No action taken.")

        if self.drop_indicator:
            df.drop(columns=outlier_columns, inplace=True)
            self.logger.info("Dropped outlier indicator columns.")

        self.logger.info("Transformation completed.")
        return df

class SubsetSelectorTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, selected_columns):
        self.selected_columns = selected_columns
        self.logger = logging.getLogger(__name__)

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        self.logger.info("Selecting specified columns...")
        if isinstance(X, pd.DataFrame):
            # Select the specified columns
            selected_df = X[self.selected_columns]
            self.logger.info("Column selection complete.")
            return selected_df
        else:
            self.logger.error("Input must be a DataFrame.")
            raise ValueError("Input must be a DataFrame.")


class DataBlendingTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, dfs, method, on=None, axis=0, how='left', ignore_index=False):
        if not isinstance(dfs, list):
            self.dfs = [dfs]
        else:
            self.dfs = dfs
        self.method = method
        self.on = on
        self.axis = axis
        self.how = how
        self.ignore_index = ignore_index
        self.logger = logging.getLogger(__name__)

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()
        self.logger.info(f"Applying data blending method: {self.method}")
        if self.method == 'Concatenate':
            df = pd.concat(self.dfs, axis=self.axis, ignore_index=self.ignore_index)
        elif self.method == 'Merge':
            for i in range(0, len(self.dfs)):
                df = pd.merge(df, self.dfs[i], on=self.on, how=self.how, suffixes=('', f'_{i}'))
        else:
            pass
        self.logger.info("Data blending complete.")
        return df