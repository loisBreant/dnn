import kagglehub
import os
import gdown

os.makedirs("data", exist_ok=True)
# Download and unzip the train dataset

os.makedirs("data/train_data", exist_ok=True)

file_id = '1GAB3uGsmAyLgtDBDONbil08vVu5wJcG3'
url = f'https://drive.google.com/uc?id={file_id}'
output_zip = 'dataset.zip'
destination_folder = 'data/train_data/'

gdown.download(url, output_zip, quiet=False)

if not os.path.exists(destination_folder):
    os.makedirs(destination_folder)

os.system(f"rm -rf {destination_folder}/*")
os.system(f"unzip -q -j {output_zip} -d {destination_folder}")
os.remove(output_zip)

print(f"Nombre d'images trouvées : {len(os.listdir(destination_folder))}")

# Download the LOL dataset from Kaggle: for testing purpose
path = kagglehub.dataset_download("soumikrakshit/lol-dataset")
print(path)
os.makedirs("data/test_data", exist_ok=True)
os.system(f"mv {path}/* data/test_data/")

# Download the Adobe FiveK dataset: for testing purpose
path_fivek = kagglehub.dataset_download("weipengzhang/adobe-fivek")
os.makedirs("data/test_data/adobe_fivek", exist_ok=True)
os.system(f"cp -r {path_fivek}/* data/test_data/adobe_fivek/")
