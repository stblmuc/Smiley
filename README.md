
CNN Architektur von https://github.com/hwalsuklee/tensorflow-mnist-cnn
Rahmen von https://github.com/sugyan/tensorflow-mnist



README von https://github.com/sugyan/tensorflow-mnist:


# MNIST classification by TensorFlow #

- [MNIST For ML Beginners](https://www.tensorflow.org/tutorials/mnist/beginners/)
- [Deep MNIST for Experts](https://www.tensorflow.org/tutorials/mnist/pros/)

![screencast](https://cloud.githubusercontent.com/assets/80381/11339453/f04f885e-923c-11e5-8845-33c16978c54d.gif)


## JUST PYTHON ##

### Requirement ###

- Python >=2.7 or >=3.4
  - TensorFlow >=1.0


### How to run ###

    $ set FLASK_APP=main.py
    $ python -m flask run



## WITH NODE (original) ## 

Caution: gunicorn does not work on Windows

### Requirement ###

- Python >=2.7 or >=3.4
  - TensorFlow >=1.0



### How to run ###

    $ pip install -r requirements.txt
   $ gunicorn main:app --log-file=-


### Deploy to Heroku ###

    $ heroku apps:create [NAME]
    $ heroku buildpacks:add heroku/nodejs
    $ heroku buildpacks:add heroku/python
    $ git push heroku master

or Heroku Button.

[![Deploy](https://www.herokucdn.com/deploy/button.svg)](https://heroku.com/deploy)
