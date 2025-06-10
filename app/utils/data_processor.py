import os
import pandas as pd
from backend.app.config import RESIZE_DIR

def load_patient_data(which_eye='od'):
    if which_eye == 'od':
        filename = 'cleaned_patient_data_od_with_images.csv'
    else:
        filename = 'cleaned_patient_data_os_with_images.csv'
    csv_path = os.path.join(RESIZE_DIR, filename)
    return pd.read_csv(csv_path)