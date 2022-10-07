import os
import pandas as pd
import keras_ocr
from configuration.config import OCR_DATA_CSV_PATH
from configuration.config import ROOT_DATA_PATH


class ModelKeras:
    def __init__(self) -> None:
        self.df = pd.read_csv(OCR_DATA_CSV_PATH)
        self.root_path = ROOT_DATA_PATH

    def get_image_list(self, images):
        for i in range(0, len(self.df)):
            ext_fold = str(self.df.loc[i, "External_Folder"])
            int_fold = str(self.df.loc[i, "Internal_Folder"])
            file_name = str(self.df.loc[i, "File_Name"])
            file_name_ext = file_name + ".jpg"
            file_path = os.path.join(path, ext_fold, int_fold, file_name_ext)
            img = keras_ocr.tools.read(file_path)
            images.append(img)
        return images

    def get_predictions(self, images):
        pipeline = keras_ocr.pipeline.Pipeline()
        res = pipeline.recognize(images)
        return res
    
    def driver(self):
        images = []
        images = self.get_image_list(images)
        res = self.get_predictions(images)
        print(len(res))


if __name__ == "__main__":
    model = ModelKeras()
    model.driver()