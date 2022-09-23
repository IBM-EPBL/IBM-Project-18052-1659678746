# IBM-Project-18052-1659678746

# VirtualEye - Life Guard for Swimming Pools to Detect Active Drowning

For now I've done a simple implementation using Python powered by YOLO v3

Current potential issues:
</br>```Lag while playback?``` 
</br>```Cause: The current implementation runs on CPU```
</br>```Solution: GPU implementation would produce higher FPS```

Future updates would include Tensorflow implementation that would be capable of using CUDA Enabled GPUs, this would enhance playback performance significantly

The current model is trained using ~250 sample images
</br><b><i>more the number of sample images, more the accuracy!</i></b>

Weights for the project can be downloaded from <u><a href="https://drive.google.com/file/d/1-ECcQYbQQvyVEwvT54T0sdTdu9R3AZkM/view?usp=sharing">here!</a></u>
</br>Step 1: Ensure that the file is named as "<b>yolov3_training_3000.weights</b>"
</br>Step 2: Paste the file in the following directory
</br>Directory: ```/weights/<paste the weight file here!>```
</br><b><i>caution: The directory and naming has to be same as given above or the project won't work!</i></b>

Get the test video at: https://youtu.be/nWOSBu4FdO0
</br>Download the video using any YouTube video downloader, ensure that you're saving it as "<b>swimming_pool1.mp4</b>"
</br>Directory: ```/media/<paste the test video here!>```
</br><b><i>caution: The directory and naming has to be same as above or the project won't work!</i></b>

<b>To run the project:</b>
<br>Step 1: Open a terminal and ```cd``` to the project directory
<br>Step 2: Execute the following command to activate the environment:
<br>```./etc/Scripts/activate```
<br>Step 3: To run the project:
<br>```python app.py``` (this launches a project instance on localhost)