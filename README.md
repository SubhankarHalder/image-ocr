# Keras OCR versus Tesseract - Subhankar Halder

# Project Brief

The objective of this project was to compare Keras-OCR and Tesseract models and infer which model performs better OCR in a given dataset. For this project, 41 images from the [KAIST Scene Text Database](http://www.iapr-tc11.org/mediawiki/index.php?title=KAIST_Scene_Text_Database) was used as OCR data. It was found that Keras-OCR is statistically better than Tesseract on the CER(Character Error Rate) metric.

# Dataset

41 images from the [KAIST Scene Text Database](http://www.iapr-tc11.org/mediawiki/index.php?title=KAIST_Scene_Text_Database) was used as OCR data.

# Scripts

The following contains a brief description about the Python Scripts and files used in this project.

  * [config](/models/config.py): Contains file/folder paths
  * [keras_ocr_model](/models/keras_ocr_model.py): Script to generate predictions using the Keras OCR Model
  * [tesseract_model](/models/tesseract_model.py): Script to generate predictions using the Tesseract Model
  * [stats_compare](/models/stats_compare.py): Does a T-Test to compare the CER values between the Keras OCR and Tesseract Models
  * [transfer_images](/models/transfer_images.py): Transfers 41 images from the raw dataset to an image folder. This image folder is only for the user to visualize the images in Github
  * [ocr_data.csv](/input_file/ocr_data.csv): File has metadata for the 41 images including its ground Truth
  * [keras_ocr.csv](/output/keras_ocr.csv): File contains the predictions of the Keras OCR model along with the CER values
  * [tesseract_ocr.csv](/output/tesseract_ocr.csv): File contains the predictions of the Tesseract model along with the CER values
  * [stats.txt](/output/stats.txt): File contains the T-Test Statistic and the Mean CER values of the 2 models
  * [requirements.txt](/requirements.txt): Python Libraries Requirements

# Results

The Keras OCR CER Mean was 0.26 and the Tesseract CER Mean was 2.12. A higher CER score means worse OCR accuracy. The p-statistic of the 2 Sample T-Test was 0.004. Thus, based on the p-values, Keras OCR performance was much better than the Tesseract model performance on the specific dataset.