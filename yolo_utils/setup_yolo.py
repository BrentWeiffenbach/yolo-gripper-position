import os
import requests
def setup_yolo(model_name):
    """
    Set up the YOLO model by checking if the specified model file exists.
    If the model file does not exist, it downloads the model from a predefined URL.
    Args:
        model_name (str): The name of the model file to check or download.
    Raises:
        Exception: If the model file cannot be downloaded.
    """
    
    #  Set up the yolo model
    # Check if model exists
    if not os.path.isfile(model_name):
        print(f'{model_name} does not exist. Downloading...')
        download_url = 'https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n-seg.pt'
        response = requests.get(download_url)

        if response.status_code == 200:
            with open(model_name, 'wb') as file:
                file.write(response.content)
            print(f'Downloaded {model_name}')
        else:
            print(f'Failed to download {model_name}')
