# Boreflix - Movie Recommendation System

## How to run the project?

1. Install `docker`

2. Clone the Repo

3. Trained model and required datasets are present in `Fastapi` directory

4. Build the docker image

   ```bash
   cd Fastapi

   docker build -t movierecommender:latest .
   ```

5. Create a container from the built image

   ```bash
   docker run -d -p 8000:80 movierecommender:latest
   ```

6. Go to your browser and type `http://localhost:8000/` in the address bar.

7. Hurray! That's it.

## How to train the model?

1. Clone the Repo

2. Create a python environment

    ```bash
    python3 -m venv movie.env
    ```

3. Activate the environment

    ```bash
    source movie.env/bin/activate
    ```

4. Install required packges in `model-requirements.txt`

    ```bash
    pip install -r model-requirements.txt
    ```

5. Unzip the datasets in dataset folder

6. Run `train.py` file

    ```bash
    python train.py
    ```

7. JupyterNotebooks are also available
