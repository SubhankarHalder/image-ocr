import scipy.stats as stats
import pandas as pd
import numpy as np
from config import KERAS_OUTPUT_CSV_PATH
from config import TESSERACT_OUTPUT_CSV_PATH
from config import STATISTICS_OUTPUT_PATH


class Stats:
    def __init__(self) -> None:
        self.df_keras = pd.read_csv(KERAS_OUTPUT_CSV_PATH)
        self.df_tess = pd.read_csv(TESSERACT_OUTPUT_CSV_PATH)
        self.output_text_file = STATISTICS_OUTPUT_PATH
    
    def get_mean(self, keras_grp, tess_grp):
        return np.mean(keras_grp), np.mean(tess_grp)
    
    def get_variance(self, keras_grp, tess_grp):
        return np.var(keras_grp), np.var(tess_grp)

    def publish_output(self, res, mean_string):
        with open(self.output_text_file, 'w') as f:
            f.write(res)
            f.write("\n")
            f.write(mean_string)
    
    def driver(self):
        keras_grp = np.array(self.df_keras["CER"].tolist())
        tess_grp = np.array(self.df_tess["CER"].tolist())
        keras_var, tess_var = self.get_variance(keras_grp, tess_grp)
        keras_mean, tess_mean = self.get_mean(keras_grp, tess_grp)
        print(keras_mean, tess_mean)
        mean_string = "Keras CER Mean = " + str(keras_mean) + " Tesseract CER Mean = " + str(tess_mean)
        var_ratio = tess_var/keras_var
        if var_ratio > 4:
            res = stats.ttest_ind(a=keras_grp, b=tess_grp, equal_var=False)
        else:
            res = stats.ttest_ind(a=keras_grp, b=tess_grp, equal_var=True)
        res = str(res)
        print(res)
        self.publish_output(res, mean_string)
        

if __name__ == "__main__":
    model = Stats()
    model.driver()
        