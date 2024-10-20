import os, shutil
import pandas as pd
import numpy as np
from custom_logger import get_logger

class convert_survey_excel_to_csv():

    def __init__(self, log_filename):
        self.logger = get_logger(log_filename)
        self.input_folder = os.path.join(os.getcwd(), r"Input")
        self.csv_folder = r"output"

    def read_data(self):
        try:
            self.verify_folder(self.csv_folder)
            self.logger.info("Removed old survey files from the output folder")
            for file in os.listdir(self.input_folder):
                if file.endswith('.xlsx') or file.endswith('.xls'):
                    df = pd.read_excel(os.path.join(self.input_folder, file), skiprows=[2,3])
                    if self.data_preprocess(df):
                        self.logger.info(f"{file} is cleaned and saved as csv chucks in output folder.")
                    else:
                        self.logger.error(f"Error processing the survey file {file}")
                elif file.endswith('.csv'):
                    df = pd.read_csv(os.path.join(self.input_folder, file))
                    df.dropna(how='all', axis=0, inplace=True)
                    df = df.reset_index(drop=True)
                    df.dropna(how='all', axis=1, inplace=True)
                    df.to_csv(os.path.join(self.csv_folder, file))
                    self.logger.info(f"{file} Csv file is cleaned and saved in output folder")
            return True
        except Exception as e:
            self.logger.error(f"function read_data : {e}", exc_info=True)
            return False

    def fill_with_previous_string_and_extra_text(self, column):
        # Forward fill NaN values and add extra text to the previous value
        for i in range(1, len(column)):
            if pd.isnull(column[i]):
                column[i] = "Percentage of " + str(column[i-1])
        return column

    def remove_punctuation(self, text):
        try:
            text = text.replace('\n', ' ')
            return text.translate(str.maketrans("", "", "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"))
        except Exception as e:
            self.logger.error(f"function remove_punctuation : {e}", exc_info=True)
            return text
    def verify_folder(self, folder_path):
        try:
            if os.path.exists(folder_path):
                # List the files in the folder
                files = os.listdir(folder_path)
                if files:
                    # Remove each file inside the folder
                    for file in files:
                        file_path = os.path.join(folder_path, file)
                        # Check if it's a file (not a directory)
                        if os.path.isfile(file_path):
                            os.remove(file_path)  # Delete the file
                        elif os.path.isdir(file_path):
                            # If it's a subfolder, you can delete it recursively using shutil.rmtree()
                            shutil.rmtree(file_path)
                else:
                    self.logger.info(f"{folder_path} Folder is empty")
            else:
                os.makedirs(folder_path, exist_ok=True)
                self.logger.info(f"Created a Folder with name {folder_path}")
            return True
        except Exception as e:
            self.logger.error(f"function verify_folder : {e}", exc_info=True)
            return e

    def data_preprocess(self, df, do=False):
        try:
            self.logger.info("Data Preprocessing started")
            pd.set_option('future.no_silent_downcasting', True)
            stop_column = 'Question 2'
            for i, col in enumerate(df.columns):
                if stop_column in col:
                    df = df.loc[:, :col]
                    break
            df = df.drop(columns=[df.columns[-1]])
            df = df.replace(r'^\s+$', np.nan, regex=True)
            df.columns = [col.strip() for col in df.columns]
            df.columns = df.iloc[0]  # Set first row as column names
            df = df.drop(df.index[0]).reset_index(drop=True)
            df.rename(columns={df.columns[1]: "Total - Both Male & Female"}, inplace=True)
            df.dropna(how='all', axis=0, inplace=True)
            df = df.reset_index(drop=True)
            df.dropna(how='all', axis=1, inplace=True)
            try:
                df['Demographics'] = self.fill_with_previous_string_and_extra_text(df['Demographics'])
            except Exception as e:
                self.logger.error(f"function fill_with_previous_string_and_extra_text : {e}", exc_info=True)
            if do:
                mask = df.drop(columns=["Demographics"]).isna().all(axis=1)
                cleaned_df = df[~mask].reset_index(drop=True)
            else:
                cleaned_df = df
            # cleaned_df.to_csv(os.path.join(self.csv_folder, f'Cleaned_data.csv'), index=False)
            self.Spliting_the_data(df)
            return True
        except Exception as e:
            self.logger.error(f"function data_preprocess : {e}", exc_info=True)
            return False

    def Spliting_the_data(self, df, ignore_column="Demographics"):
        try:
            self.logger.info(f"Splitting the data into CSV chunks")
            question_df = pd.DataFrame()
            # Create a mask to identify rows where all columns except 'ID' are NaN
            mask = df.drop(columns=[ignore_column]).isna().all(axis=1)

            # Split DataFrame based on the mask
            split_indices = df.index[mask].tolist()  # List of indices where all other columns are NaN
            split_indices.append(len(df))  # Add the last row index to handle the final slice

            # Split DataFrame into sub-DataFrames
            sub_dfs = []
            start_idx = 0
            for end_idx in split_indices:
                sub_df = df.iloc[start_idx:end_idx].reset_index(drop=True)
                sub_df.dropna(how='any', axis=1, inplace=True)
                if not sub_df.empty:
                    sub_dfs.append(sub_df)
                start_idx = end_idx + 1  # Move to the next starting index
            ids_with_true_mask = df.loc[mask, ignore_column].to_list()
            ids_with_true_mask.remove("Questions")
            ids_with_true_mask.insert(0, "Gender")
            for i, sub_df in enumerate(sub_dfs):
                filename = ids_with_true_mask[i][:12].replace(" ", "_")
                print(filename)
                cleaned_text = self.remove_punctuation(ids_with_true_mask[i])
                if "Question" in cleaned_text:
                    sub_df.columns = ["Question or Query"] + list(sub_df.columns[1:])
                    sub_df["Question or Query"] = sub_df["Question or Query"].apply(
                        lambda x: cleaned_text[12:] + " : " + str(x))
                    question_df = pd.concat([question_df, sub_df], axis=0, ignore_index=True)
                else:
                    sub_df.columns = [cleaned_text] + list(sub_df.columns[1:])
                    # sub_df[cleaned_text] = sub_df[cleaned_text].apply(lambda x: cleaned_text + " : " + str(x))
                    if os.path.exists(os.path.join(self.csv_folder, ids_with_true_mask[1])):
                        sub_df.to_csv(os.path.join(self.csv_folder, ids_with_true_mask[1], f'{filename}.csv'), index=False)
                    else:
                        os.makedirs(os.path.join(self.csv_folder, ids_with_true_mask[1]))
                        sub_df.to_csv(os.path.join(self.csv_folder, ids_with_true_mask[1], f'{filename}.csv'), index=False)
            question_df["Question or Query"] = question_df["Question or Query"].apply(self.remove_punctuation)
            if os.path.exists(os.path.join(self.csv_folder, ids_with_true_mask[1])):
                question_df.to_csv(os.path.join(self.csv_folder, ids_with_true_mask[1], f'Questions.csv'), index=False)
            else:
                os.makedirs(os.path.join(self.csv_folder, ids_with_true_mask[1]))
                question_df.to_csv(os.path.join(self.csv_folder, ids_with_true_mask[1], f'Questions.csv'), index=False)
            return ids_with_true_mask
        except Exception as e:
            self.logger.error(f"function data_preprocess : {e}", exc_info=True)
            return e

    # def df_for_graphs(self):
    #     graph_dict = {}
    #     for dirpath, dirnames, filenames in os.walk(self.csv_folder):
    #         for file in filenames:
    #             if file == "Age.csv":
    #                 age_df = pd.read_csv(os.path.join(dirpath, file), skiprows=[1,2])
    #                 percentage_age_rows = age_df[age_df['Age'].str.contains('Percentage', case=False)].iloc[:, [0,2,3]]
    #                 non_percentage_age_rows = age_df[~age_df['Age'].str.contains('Percentage', case=False)].iloc[:, [0,2,3]]
    #                 graph_dict[f"{os.path.split(dirpath)[1]}_percentage_age_rows"] = percentage_age_rows
    #                 graph_dict[f"{os.path.split(dirpath)[1]}_non_percentage_age_rows"] = non_percentage_age_rows
    #                 # self.logger.info(graph_dict)
    #     return graph_dict