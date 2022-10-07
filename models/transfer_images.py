import os
import shutil
import pandas as pd
from tqdm import tqdm
from config import OCR_DATA_PATH
from config import ROOT_DATA_PATH
from config import IMAGES_PATH


class Transfer:
    def __init__(self) -> None:
        self.output_images = IMAGES_PATH
        self.root_path = ROOT_DATA_PATH
        self.df = pd.read_csv(OCR_DATA_PATH)
    
    def copy_images(self):
        for index in tqdm(range(0, len(self.df))):
            ext_fold = str(self.df.loc[index, "External_Folder"])
            int_fold = str(self.df.loc[index, "Internal_Folder"])
            file_name = str(self.df.loc[index, "File_Name"])
            file_name_ext = file_name + ".jpg"
            file_path = os.path.join(self.root_path, ext_fold, int_fold, file_name_ext)
            shutil.copy2(file_path, self.output_images)
    
    def driver(self):
        if os.path.isdir(self.output_images):
            shutil.rmtree(self.output_images)
        os.mkdir(self.output_images)
        self.copy_images()


if __name__ == "__main__":
    model = Transfer()
    model.driver()
         
