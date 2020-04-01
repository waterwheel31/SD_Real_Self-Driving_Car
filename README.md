# Programming a Real Self Driving Car Project



<p align="center">

[![Youtube](http://img.youtube.com/vi/9qB6OQQb_co/0.jpg)](https://www.youtube.com/watch?v=9qB6OQQb_co "Capstone")

  (click to see Youtube Video)
</p>


## Objective 

  - Control a car based on ROS (Robotic Operation System), using information from traffoc light classificaiotn 
  - This runs on a simulator and a real car

## Approach/Process 

- The structure of ROS is as below 

<img src="final-project-ros-graph-v2.png" width=500>

- Traffic light classification model (this is used in the "Traffic Light Detection Node" in the map above) uses pre-trained model: 
  - The pretrained model is SSD-Mobile Net V1 (from TensorFlow Models)
  - Training data is traffic light images in the simualtor, annotated by Vatsal (https://github.com/coldKnight/TrafficLight_Detection-TensorFlowAPI#get-the-dataset)


## Result 

- The car could run smoothly in the simulator (see the video above)