a really old and ugly neural network made for number classification from black and white 30x34 images 

It requires:
* Python3
* Tensorflow
* Tflearn
* Opencv2
* Numpy
* PIL
* Imutils

to use it specify a path to you image on *line 34* in **nn6_p.py**
```python
X = [imToTensor(imToAr("t/9a"))]
#                       ^^^^change this here to your image path
```
then run **nn6_p.py**
```python
python3 nn6_p.py
```

if you want to re-train it you can try to poke around **nn6.py** but i recommend making your own trainer, nn is defined from *line 35* to *line 46* in **nn6.py** or **nn6_p.py**

**net7.tflearn.data-00000-of-00001**, **net7.tflearn.index** and **net7.tflearn.meta** is model and weight data used to load the neural network

# **this code is more then a year old!**
