an old, ugly and badly designed cnn for number recognition from live video

It requires:
* Python3
* Tensorflow
* Tflearn
* Numpy
* Opencv2
* PIL

to use it run **cnn8_p.py** you need a camera connected
```python
python3 cnn8_p.py
```
to change camera/video sourse modify *line 106* in **cnn8_p.py**
```python
cum = cv2.VideoCapture(0)
#                      ^this thing here
```
to re-train it i adwise making your own trainer, in **cnn8_p.py** cnn is defined from *line 59* to *line 76*, the cnn is made for black and white 30x34 images

just don't forget to
```python
import tflearn
```