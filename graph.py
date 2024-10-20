import pandas as pd
import matplotlib.pyplot as plt
import io, os
from custom_logger import get_logger

class plots:
    def __init__(self, log_filename, folder_path=r"Graph_data"):
        self.logger = get_logger(log_filename)
        self.folder_path = folder_path
        self.figure, self.axis = plt.subplots(3*len(os.listdir(folder_path)), 2, figsize=(12, 12*len(os.listdir(folder_path))))

    def read_and_partition_gender(self, file_path):
        try:
            filename = os.path.split(file_path)[1]
            df = pd.read_csv(file_path)  # Replace with your CSV file path
            percentage_rows = df[df[filename.split(".")[0].replace("_"," ")].str.contains('Percentage', case=False)].iloc[:, [0, 2, 3]]
            percentage_rows = percentage_rows.drop(index=percentage_rows.index[0]).reset_index(drop=True)
            non_percentage_rows = df[~df[filename.split(".")[0].replace("_"," ")].str.contains('Percentage', case=False)].iloc[:, [0, 2, 3]]
            non_percentage_rows = non_percentage_rows.drop(index=non_percentage_rows.index[0]).reset_index(drop=True)
            return percentage_rows, non_percentage_rows
        except Exception as e:
            self.logger.error(f"error in funciton read_and_partition_gender : {e}")
            return "Error", e


    def bar_plot(self, df, row_pos, col_pos, x_axis_column, y_axis=["Male", "Female"]):
        try:
            df[y_axis] = df[y_axis].apply(pd.to_numeric, errors='coerce')
            index = range(len(df[x_axis_column]))
            self.axis[row_pos][col_pos].bar([p + 0.25 for p in index], df[y_axis[0]], width=0.25, color='#1f77b4',
                                 label=f'{y_axis[0]}',
                                 edgecolor='black')
            self.axis[row_pos][col_pos].bar([p + 0.25 * 2 for p in index], df[y_axis[1]], width=0.25,
                                 color='#ff7f0e', label=f'{y_axis[1]}',
                                 edgecolor='black')
            for bar in self.axis[row_pos][col_pos].containers:
                for rect in bar:
                    height = rect.get_height()
                    self.axis[row_pos][col_pos].annotate(f'{height}',
                                              xy=(rect.get_x() + rect.get_width() / 2, height),
                                              xytext=(0, 3),  # 3 points vertical offset
                                              textcoords="offset points",
                                              ha='center',
                                              va='bottom', fontsize=10)
            self.axis[row_pos][col_pos].set_title(f"{x_axis_column} and Gender Distribution")
            self.axis[row_pos][col_pos].set_xlabel(f"{x_axis_column} Group")
            self.axis[row_pos][col_pos].set_ylabel("Count")
            self.axis[row_pos][col_pos].set_xticks(range(len(df[x_axis_column])))
            self.axis[row_pos][col_pos].set_xticklabels(df[x_axis_column].to_list(), rotation=45)
            self.axis[row_pos][col_pos].legend()
            return True
        except Exception as e:
            self.logger.error(f"error in funciton Bar_plot : {e}")
            return False

    def pie_chart(self, df, row_pos, col_pos, x_axis_column, y_axis=["Male", "Female"]):
        try:
            labels = [value.strip('Percentage of') for value in df[x_axis_column]]
            sizes = [float(value.strip('%')) for value in df[y_axis[0]]]
            explode = (0.1,) + (0,) * (len(sizes) - 1)
            self.axis[row_pos][col_pos].pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%', shadow=True, startangle=140)
            self.axis[row_pos][col_pos].set_title(f"{y_axis[0]} Population Distribution")
            self.axis[row_pos][col_pos].axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
            row_pos += 1

            sizes = [float(value.strip('%')) for value in df[y_axis[1]]]
            explode = (0.1,) + (0,) * (len(sizes) - 1)
            self.axis[row_pos][col_pos].pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%', shadow=True, startangle=140)
            self.axis[row_pos][col_pos].set_title(f"{y_axis[1]} Population Distribution")
            self.axis[row_pos][col_pos].axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
            return True
        except Exception as e:
            self.logger.error(f"error in funciton pie_chart : {e}")
            return False

    def plot_grpahs(self):
        try:
            for axis_cnt, data_folder in enumerate(os.listdir(self.folder_path)):
                row_pos = 0
                for file in os.listdir(os.path.join(os.getcwd(), self.folder_path, data_folder)):
                    if "Age.csv" == file:
                        percentage_data, non_percentage_data = self.read_and_partition_gender(file_path=os.path.join(os.getcwd(),self.folder_path, data_folder, "Age.csv"))
                        if self.bar_plot(df=non_percentage_data, row_pos=row_pos, col_pos=axis_cnt, x_axis_column="Age"):
                            row_pos += 1
                        if self.pie_chart(df=percentage_data, row_pos=row_pos, col_pos=axis_cnt, x_axis_column="Age"):
                            row_pos += 2

                    elif data_folder.split()[0] in file.replace("_", " "):
                        percentage_data, non_percentage_data = self.read_and_partition_gender(file_path=os.path.join(os.getcwd(),self.folder_path, data_folder, file))
                        if self.bar_plot(df=non_percentage_data, row_pos=row_pos, col_pos=axis_cnt, x_axis_column=data_folder):
                            row_pos += 1
                        if self.pie_chart(df=percentage_data, row_pos=row_pos, col_pos=axis_cnt, x_axis_column=data_folder):
                            row_pos += 2

            # Save the pie chart to a BytesIO object
            img = io.BytesIO()
            plt.tight_layout()  # Adjusts the subplots to fit into the figure area.
            plt.savefig(img, format='png')
            plt.close(self.figure)  # Close the figure after saving to free up memory
            img.seek(0)
            self.logger.info("Graphs generated successfully")
            return img
        except Exception as e:
            self.logger.error(f"error in funciton plot_grpahs : {e}")
            return False
