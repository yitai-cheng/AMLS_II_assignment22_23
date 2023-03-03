import os
import zipfile

import requests as requests


def run_fetch():
    file_url = "https://storage.googleapis.com/kaggle-competitions-data/kaggle-v2/29653/2420395/bundle/archive.zip?GoogleAccessId=web-data@kaggle-161607.iam.gserviceaccount.com&Expires=1678040419&Signature=pVgx%2BGFn6xdgQ5%2BtokFa%2FwL9e5eylZ6d7aWQEUgmC5EMOkt88dPdfAetdIP0ZsITt7v7qW6%2BcR1DvDGbchYORwUZH558hU5GNy0iOdh9Pg%2FD%2FZ4jxKCCIUjJRhSejwHsKNgEdxeuY%2F31s0pYnmAgB%2FqaZDS9TB078nSb6kpe4XA9zPIs%2FWSD3rYOP8lteHE2PEGJZgUT6nqmI0Odb4cQieDbkr4RWymyl3OK6gO6m%2FnzAxSm2As%2BuR5mnsHSgaVFuWlRkzpglifMFHxnkLx8GcsYxvc5cZn%2BQDb6NvKkzaGcgAOBbYwCNqYcdw7CsqdYnPn5WfVbAyqqM%2FrPtuo5Ew%3D%3D&response-content-disposition=attachment%3B+filename%3Drsna-miccai-brain-tumor-radiogenomic-classification.zip"
    filename = os.path.join(os.getcwd(), "data.zip")
    r = requests.get(file_url, stream=True)

    with open(filename, "wb") as f:
        for chunk in r.iter_content(chunk_size=1024):

            # writing one chunk at a time to pdf file
            if chunk:
                f.write(chunk)

    # Make a directory to store the data.
    os.makedirs("Data")

    # Unzip data in the newly created directory.
    with zipfile.ZipFile("data.zip", "r") as z_fp:
        z_fp.extractall("./Data/")

