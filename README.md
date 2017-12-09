Artificial Neural Networks in Python
===

This was written as ANN learning experience project, and is by no means a super efficient ANN module (yet).
It currently can implement a perceptron model or a ANN with a single hidden layer.

# For Best Results

Use the weights file `weights_perceptron.npy` if using the perceptron network.
Use the weights file `weights_multilayer.npy` if using the multilayer network.

# Dependencies
Requires the numpy module and python 3.6 or newer. This can be installed with pip:

```
> pip install numpy
```

If you don't have pip, here are details on installing numpy: https://scipy.org/install.html

# How to Use
```
usage: pyann.py [-h] [--weights WEIGHTS]
                   {train,test} {perceptron,multi-layer}

positional arguments:
  {train,test}          command to run
  {perceptron,multi-layer}
                        what type of ANN to run

optional arguments:
  -h, --help            show this help message and exit
  --weights WEIGHTS     load a weights file instead of rng weights

```
note: Some settings such as learning rate, train/test file names are located at the top of project2.py as constants.

### Data File Format

Data files must be tab delimited and in the format `<latitude> <longitude> <class>`
