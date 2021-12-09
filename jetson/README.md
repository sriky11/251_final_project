## Model Creation:

Inside the "/Drowsiness/Model" folder our model called "BlinkModel.t7" already exists, which is the one that came from DBSE-monitor github project (https://github.com/altaga/DBSE-monitor/tree/master/Drowsiness/Model)". The one we trained is currently being used (with our own eyes added to the database).

The training has the following parameters as input.

- input image shape: (24, 24)
- validation_ratio: 0.1
- batch_size: 64
- epochs: 40
- learning rate: 0.001
- loss function: cross entropy loss
- optimizer: Adam

In the first part of the code you can modify the parameters, according to your training criteria.

# How to run
In a jetsan, run "build.sh" to build the dockerfile. The dockerfile will create a new user, and this user will be set with audio so that the docker container's audio will use the Jetsan's audio. We use the latest (as of 12/2021) NVidia ML container for Jetscan that has pytorch, jupyter, etc.

Then run "host_runner.sh" to start the container with the proper options to allow webcam access and audio access.

Then, you can run "jupyter notebook --ip 0.0.0.0" and access the notebook from a browser. After running the notebook, if changes are made, you may need to rerun it more than once, as the first rerun will result in a failure since the camera resources are not properly releated (the code just runs in an infinite loop, reading from the webcam).


# How does it work:

Let's go through a revision of the algorithms and procedures of both CV systems (Drowsiness and alert on one side and Blind spot detection on the other). The installation is remarkably easy as I have already provided an image for the project.

ALL the code is well explained in "Notebook.ipynb" file.

Please take a look at it for extensive explanation and documentation.

The sleep monitor uses the following libraries:

- OpenCV:
- Image processing. 
    - (OpenCV) Haarcascades implementation. 
    - (OpenCV) Blink eye speed detection.
    - (Pytorch) Eye Status (Open / Close)
- VLC: 
    - Player sound alert.

Only in Jetson Nano:

- Smbus:
    - Accelerometer reading.
- Twilio:
    - Emergency notification delivery.
- Requests:
    - Geolocation

The flicker detection algorithm is as follows:

- Detection that there is a face of a person behind the wheel:

<img src="https://i.ibb.co/ZMvwvfp/Face.png" width="600">

- If a face is detected, perform eye search.

<img src="https://i.ibb.co/StK0t2x/Abiertos.png" width="600">

- Once we have detected the eyes, we cut them out of the image so that we can use them as input for our convolutional PyTorch network.

<img src="https://i.ibb.co/0FYT0DN/Abiertoss.png" width="600">

- The model is designed to detect the state of the eyes, therefore it is necessary that at least one of the eyes is detected as open so that the algorithm does not start generating alerts, if it detects that both eyes are closed for at least 2 seconds , the alert will be activated, since the security of the system is the main thing, the algorithm has a second layer of security explained below.

- Because a blink lasts approximately 350 milliseconds then a single blink will not cause problems, however once the person keeps blinking for more than 2 or 3 seconds (according to our criteria) it will mean for the system that the person is falling asleep. Not separating the eyes from the road being one of the most important rules of driving.

<img src="https://i.ibb.co/kQ12W79/alert1.png" width="600">
<img src="https://i.ibb.co/LdVD7v2/alert2.png" width="600">

- Also during the development I found a incredible use case, when one turns to look at his cell phone, the system also detects that you are not seeing the road. This being a new aspect that we will be exploring in future versions of the system to improve detection when a driver is not looking at the road and is distracted. But, for now it is one of those unintended great finds.

<img src="https://i.ibb.co/mHZ4VdX/Cel.png" width="600">
<img src="https://i.ibb.co/3k512YS/cel2.png" width="600">

Whether it's because of the pytorch convolutional network model or the Haarcascades, the monitor will not allow you to take your eyes off the road, as it is extremely dangerous to do that while driving.

<img src="https://i.ibb.co/D84YbYb/Whats-App-Image-2020-03-16-at-12-35-40.jpg" width="600">


# Projects/code this project borrows from:

https://github.com/altaga/DBSE-monitor/tree/master/Drowsiness

https://github.com/TheBiggerGuy/docker-pulseaudio-example