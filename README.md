# IBM-Project-18052-1659678746

# VirtualEye - Life Guard for Swimming Pools to Detect Active Drowning

For now I've done a simple implementation using Python powered by YOLO v3

The reason why there's lag while playback is because the current implementation runs on CPU

Future updates would include Tensorflow implementation that would be capable of using CUDA Enabled GPUs that would enhance playback performance significantly

The current model is trained using ~250 sample images
<b><i>more the number of sample images, more the accuracy!</i></b>

Weights for the project can be downloaded from <u><a href="https://drive.google.com/file/d/1-ECcQYbQQvyVEwvT54T0sdTdu9R3AZkM/view?usp=sharing">here!</a></u>
Step 1: Ensure that the file is named as "yolov3_training_3000.weights"
Step 2: Paste the file in the following directory
Directory: /weights/&lt;paste the weight file here!&gt;
<b><i>caution: The directory and naming has to be same as given above or the project won't work!</i></b>

Get the test video at: https://youtu.be/nWOSBu4FdO0
Step 1: Download the video using any YouTube video downloader
Step 2: save it as "swimming_pool1.mp4"
Directory: /media/&lt;paste the test video here!&gt;
<b><i>caution: The directory and naming has to be same as above or the project won't work!</i></b>
