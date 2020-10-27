# FireFactory
Lofi generator (Fire beats for that quick fast cash yo)

## Requirements
* [Python 3.8](https://www.python.org/downloads/) (pip3)
* [Node v12.19.0](https://nodejs.org/en/download/) (npm version 6.14.8)

## Setup

#### Virtual Environment Setup (Optional)
 
1) Create env
```sh
python3 -m venv .env
```
> to install python venv run ```sudo apt install python3-venv```

2) activate environment
```sh
source .env/bin/activate
```

#### Install Dependencies
1) Python Dependencies for the App and the Jupyter Notebook
```sh
pip3 install -r requirements.txt
```

3) Node dependencies for running the App
```sh
npm install
```
> Needs to be called from the `src/frontend` directory


## Run the Application
1) Start the Python API backend
```sh
python3 -m src.api
```
> Run from the root directory

2) Start the frontend web client (in a seperate terminal)
```sh
npm start
```
> Run from the `src/frontend` directory

## Run Notebook
#### JupyterLab
Run `jupyter lab` and select `notebook.ipynb`