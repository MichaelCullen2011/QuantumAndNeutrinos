# NST Backend

## Description

This is the Flask server that runs the Neural Style Transfer transformations through a Heroku host.

## Getting Started

### Installing

To look at the code just fork this repo and set up a virtual environment and install requirements.txt using
```
python -m venv .venv
.venv\Scripts\activate.bat
pip install -r 'requirements.txt'
```

### Executing program nst program
The NST program uses Tensorflow and Keras to train a model in the style of a specific picture (in our case a painting). This model can then take an input picture and style it in the form of the trained model (painting) resulting in some nice edited pictures.

```
python nst_lite.py
```
![alt text](https://github.com/MichaelCullen2011/NSTBackend/blob/main/nst_example.png?raw=true)

### The Flask program
This is largely run through app.py and handles POST and GET requests, collecting the relevant information, and performing the transform on the necessary pieces.

The NSTApp project at https://github.com/MichaelCullen2011/NSTApp uses this server to perform transforms and recieve the image.

## Authors

Contributors names and contact info

ex. Michael Cullen
michaelcullen2011@hotmail.co.uk

