## Installation

EMBA is built upon [ROS](http://www.ros.org/).
The installation instructions of ROS can be found [here](http://wiki.ros.org/noetic/Installation/Ubuntu).
We have tested this software on Ubuntu 20.04 and ROS noetic.

Install [catkin tools](http://catkin-tools.readthedocs.org/en/latest/installing.html), [vcstool](https://github.com/dirk-thomas/vcstool):

    sudo apt install python3-catkin-tools python3-vcstool

Install additional libraries:

    sudo apt install ros-noetic-image-geometry ros-noetic-camera-info-manager ros-noetic-image-view

Create a new catkin workspace (e.g., `emba_ws`) if needed:

    mkdir -p ~/emba_ws/src && cd ~/emba_ws/
    catkin config --init --mkdirs --extend /opt/ros/noetic --merge-devel --cmake-args -DCMAKE_BUILD_TYPE=Release

Download the source code:

    cd ~/emba_ws/src/
    git clone https://github.com/tub-rip/emba

Clone dependencies:

    vcs-import < emba/dependencies.yaml

Build the ROS package:

    cd ~/emba_ws
    catkin build emba
    source devel/setup.bash
