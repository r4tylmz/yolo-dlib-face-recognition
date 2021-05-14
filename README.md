## Install mandatory packages
``` 
pip install dlib
pip install face_recognition
pip install imutils
pip install opencv-python
pip install pickle
```
_In my opinion, if you want to install the libs that were defined above, you'd better install these on Ubuntu._
> Note that, if you want to run these scripts on your external Nvidia GPU, you need to compile `dlib` and `opencv-python` packages from scratch. If you're going to run these scripts on CPU, you have to change `DETECTION_METHOD` constant to `hog`

<hr>

## 1 - Generating a face dataset
Make sure you already have a python3, and you need to attach `--name --lastName --phoneNumber` arguments after the .py file.

`python gen_face_dataset.py --name staffName --lastName staffSurname --phoneNumber staffPhone`

`Press the "S" key, it saves the current frame to face_dataset/{full_name}_{staffId}`

After you've done this, your face dataset will be saved under `face_dataset/{full_name}_{staffId}` path.
Also, the staff credentials will be sent to the database through an **ASP.NET Core API**.

`HTTP POST: https://localhost:5001/api/Staff`

## 2 - Encoding faces to 128-d vectors
Before running this script, make sure your face_dataset folder and this script are in the same directory.

`python encode_faces.py`

 After that, an encodings.pickle file will be saved to your directory, where the face dataset folder is located.


## 3 - Detecting, Recognizing and Tracking
Detecting staff heads and recognizing their faces along with all the frames is quite the same.
Firstly, we have to detect and recognize all faces in the frame so that we can get all the faces recognizer centroids. After getting the recognizer centroids, we have to detect staff heads in the frame then extract their centroids too by using a pre-trained YOLOv4 model.

The complex part begins here :) We have to examine both the recognizer centroids and the Yolo centroids to calculate the minimum distance between the centroids. If a recognizer centroid is close to a Yolo centroid then we have to grab the ID number that is associated with the recognizer centroid.
We do the same steps for all centroids. Then, we have to link the ID numbers to the Yolo centroids therefore a text which contains the ID number is placed on the centroid center. So, we get this photo:

<img src="https://i2.paste.pics/861b7f4d8bb1463c801711586046edc6.png" width="1000" height="600" alt="Screenshot">

> The steps written above are done for all cameras and the tracking process is done by YOLOv4.

Starting tracking staff you need to type this on your terminal. The algorithm will try to access all possible cams in your system.

`python face_tracker.py`

## 4 - Storing Staff Activities
If a yolo centroid exceeds the line border of the frame, the ID number which belongs to the centroid and the data would be sent to the server through the API.

    data = { 
	    StaffId, 
	    RoomId, 
	    EntryTime, 
	    ExitTime}

`HTTP POST: https://localhost:5001/api/StaffActivity`
<hr>

_Thanks to PyImageSearch and, Adrian Rosebrock._
