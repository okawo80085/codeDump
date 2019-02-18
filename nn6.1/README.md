an improved version of nn6, a number classifier for black and white 30x34 images

It requires:
* Python3
* Tensorflow
* Tflearn
* Opencv2
* Numpy

to use it change *line 54* to path to you image in**nn6.1_p.py**
```python
dataPaths = ['data/0.jpg']
#            ^^^^^^^^^^^^change this to your image path
```
and run **nn6.1_p.py**
```python
python3 nn6.1_p.py
```

to re-train it replace or add files in the **data/** folder with black and white 30x34 images of numbers with labels in their names, it should look something like this:
```
111.jpg
22.png
3.jpg
9999.jpeg
```
**0-9** are labels for numbers, don't mix different labels together

to run the training process run **nn6.1.py**
```python
python3 nn6.1.py
```


**data/** does not contain the data it was trained on, it is there only as an example!

**net7.tflearn.data-00000-of-00001**, **net7.tflearn.index** and **net7.tflearn.meta** are model and weight data used to load the pre-trained model
