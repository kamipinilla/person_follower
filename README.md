## Setup

You must have ROS Kinetic installed, which is only available on Ubuntu 16.04. If you haven’t done so, create and build a catkin workspace:

```shell
mkdir -p ~/catkin_ws/src
cd ~/catkin_ws/
catkin_make
```

## Build

Clone this project into the `~/catkin_ws/src` folder. Then, build the package by running

```shell
cd ~/catkin_ws
catkin_make
```

Make sure you run the commands `source /opt/ros/kinetic/setup.bash` and `source ~/catkin_ws/devel/setup.bash` on every new shell to have access to the ROS environment and to this package (you can add this commands to the `.bashrc` file for convinience).

## Virtual environment

You need to create a Python 2 virtual environment. Create it with

```shell
pip install virtualenv
cd ~/catkin_ws/src/person_follower
virtualenv -p path/to/python2 venv
```

Make sure you specify the path to your custom installation of Python 2. If you don’t, you'll end up using the system version of Python. Activate the virtual environment with 

```shell
source venv/bin/activate
```
You can check the python version with `python -V`. Finally, install the appropriate python packages with

```shell
pip install -r requirements.txt
```

## Execution

Start the ROS core service with

```shell
roscore
```

In a new terminal, run the following command to setup and start the RosAria node:

```shell
roslaunch person_follower robot_params.launch
```

In a new terminal, run the following command to start the main node of this package:

```shell
rosrun person_follower robot_navigator.py
```

You will be shown the image received from the camera. Make the predefined posture for the robot to recognize you and start following you. 
