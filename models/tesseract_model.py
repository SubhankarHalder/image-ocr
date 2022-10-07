import os
import pandas as pd
import cv2
from tqdm import tqdm
import pytesseract
import torchmetrics
from config import OCR_DATA_PATH
from config import ROOT_DATA_PATH
from config import TESSERACT_OUTPUT_CSV_PATH
from config import TESSERACT_MODEL_PATH


class ModelTesseract:
    def __init__(self) -> None:
        self.df = pd.read_csv(OCR_DATA_PATH)
        self.root_path = ROOT_DATA_PATH
        self.output_file_path = TESSERACT_OUTPUT_CSV_PATH
        self.custom_config = r'--oem 3 --psm 11'
        self.output_df = pd.DataFrame(columns=["File_Name", "Ground_Truth_LowerCase", "Pred", "CER"])
        self.possible_set = {'1', '2', '3', '4', '5', '6', '7', '8', '9',
                             '0', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h',
                             'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q',
                             'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z'}

    def extract_values(self, index):
        ext_fold = str(self.df.loc[index, "External_Folder"])
        int_fold = str(self.df.loc[index, "Internal_Folder"])
        file_name = str(self.df.loc[index, "File_Name"])
        grt_truth_val = str(self.df.loc[index, "Ground_Truth"])
        file_name_ext = file_name + ".jpg"
        file_path = os.path.join(self.root_path, ext_fold, int_fold, file_name_ext)
        return grt_truth_val, file_name, file_path
    
    def post_processing(self, text):
        res = ""
        for val in text:
            if val in self.possible_set:
                res += val
        return res
        
    def get_predictions(self):
        pytesseract.pytesseract.tesseract_cmd = TESSERACT_MODEL_PATH
        for i in tqdm(range(0, len(self.df))):
            row = []
            grt_truth_val, file_name, file_path = self.extract_values(i)
            grt_truth_val = grt_truth_val.lower()
            grt_truth = [grt_truth_val]
            row.append(file_name)
            row.append(grt_truth_val)
            img = cv2.imread(file_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            text = pytesseract.image_to_string(img, config=self.custom_config, lang="eng")
            text = text.replace("\n", " ")
            text = text.lower()
            text = self.post_processing(text)
            output_str = [text]
            row.append(output_str[0])
            metric = torchmetrics.CharErrorRate()
            metric_val = round(metric(output_str, grt_truth).item(), 2)
            row.append(metric_val)
            self.output_df.loc[len(self.output_df)] = row
    
    def dump_file(self):
        print(self.output_df.head(10))
        self.output_df.to_csv(self.output_file_path, index=False)
    
    def driver(self):
        self.get_predictions()
        self.dump_file()


if __name__ == "__main__":
    model = ModelTesseract()
    model.driver()