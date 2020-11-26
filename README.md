# Real-Time-Face-Mask-Detector-Using-Sockets
We developed Face mask detector using the Object detection algorithm and developed a monitoring tool to transmit live streams to the server to detect those without masks.

## General Flow
![General Flow](/static/general-flow.jpg)

## Face Mask Detector Flow
![Face Mask Detector Flow](/static/yolo-mask-algo.png)

## Output

![Sender](/static/sender-image.png)

![Receiver 1](/static/1.png)

![Receiver 2](/static/2.png)

![Receiver 3](/static/3.png)

## Mask Detector weight file
Download weights file for mask detector from my google drive link :- [Mask Detector weight file](https://drive.google.com/file/d/14Ipi7mf-om2hQO-ySY3t5W7urQls1gRw/view?usp=sharing)

## Instructions to run code
1. Clone the repository on your machine
2. Dowload weights file from link given above and add it to the repository folder.
3. Change username and password in .env file in repository according to you.
4. Run command prompt in the repository folder.
4. Download all the requirements by running command `pip install -r requirements.txt`
5. Run app.py using command `python app.py`
6. Open any browser and visit http://127.0.0.1/sender and allow camera access.
7. Now visit http://127.0.0.1/receiver and check your result.
