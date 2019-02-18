an improved version of cnn8, a number classifier for live video

It requires:
* Python3
* Tensorflow
* Tflearn
* PIL
* Opencv2
* Numpy

to use it run **cnn8.1_p.py**
```python
python3 cnn8.1_p.py
```

to change camera/video source change *line 57* in **cnn8.1_p.py**
```python
cum = cv2.VideoCapture(0)
#                      ^this here
```

to re-train it replace or add files in the **data/** folder with black and white images of numbers and not numbers with labels in their names, it should look something like this:
```
111.jpg
22.png
zzz.jpeg
3.jpg
z.jpg
```
**0-9** are labels for numbers and **z** represents non-numbers, don't mix different labels together

to run the training process run **cnn8.1.py**
```python
python3 cnn8.1.py
```

**data/** does not contain the data it was trained on, it is there only as an example!

**conv_nn8.1.data-00000-of-00001**, **conv_nn8.1.index** and **conv_nn8.1.meta** are model and weight data used to load the pre-trained model
