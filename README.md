# CobotWeld


# <p align="center"> Chinese National Engineering Research Centre for Steel Structure (Hong Kong Branch) </p>
# <p align="center"> 国家钢结构结构工程技术研究中心（香港分中心）</p>
1 February 2023. (Friday)

# General Introduction
This is a CNERC project. The purpose of this project.

# Implementation details
Structure of the system
CoboWeld is implemented using the Universal Robots UR5 robotic arm as the primary action provider.

The whole system software is implemented on the ROS platform. It is ROS Noetic on Ubuntu 20.04. The software is a mixture in C++ and Python 3.8.

Visual images are provided by Intel Realsense D435i RGBD camera.

# Installation of the Universal Robots ROS Driver
The UR5 robotic arm is driven by ROS via the Universal Robots ROS Driver. It is provided via this repository in Github.

# Visualisation via RViz of ROS - a Digital Twin of the system
ROS provides a system visualisation tool, RViz. Rviz takes a description of the robot, from now on it will be called a manipulator. This description is in a URDF (Universal Robot Description Format) file.

Universal Robots provided this description file in fmauch_universal_robot/ur_description/urdf/ur5.xacro .

CoboWeld itself has a sub-package, coboweld_support, that provide all the supporting facilities the system needs. In coboweld_support, there is a urdf folder. There is an integrated urdf file. Integrated in this coboweld435.xacro file are the working environment, that is the world, the welding table, and the tools. The tools are, the welding torch and the sensor, that is the RGB-D camera - Intel Realsense D435i.

# The MoveIt package of ROS
On the ROS platform there is a package, MoveIt, that helps to plan and move the manipulator UR5, in both the virtual world in RViz and in the real world.

Before it can be used it must be installed and configured. It also provides a moveit_setup_assistant to do the setup for it.

Afterwards, it will generate a package, in the case of coboweld, coboweld_moveit_config.This is a project of the CNERC
## Chinese National Engineering Research Centre for Steel Structure (Hong Kong Branch)


It is based on previous work by Jeffery Zhou and Maggie.
That was almost completely in Python.

It is going to be partly ROS C++ and mostly in python.
It will also be heavily depend on Open3D.

It is going to use the URx (Python) for actually driving
the robotic arm UR5 (maybe UR10 later) instead of using
MoveIt of ROS.

It is being constructed on this 19 July, 2022. (Tuesday).

### Find orientation of Waypoints on the Welding Path

---

After I have upgraded from melodic to noetic, it seems 
I need to do some more for the Intel Realsense. The 
details are [here](https://answers.ros.org/question/363889/intel-realsens-on-ubuntu-2004-ros-noetic-installation-desription/).   

I will have to do this anyway if I want to use Realsense, 
especially if I want to do hand eye calibration. 

Just done the install, reboot and test. ROS and Unistall still 
need to be done.

---

# Installation of this package

RobotWeld depends on:
1. C++, Python3, UR Robot Driver
2. It uses URx to control the UR5 or R10. Therefore, it is necessary to install URx. It is now much simpler than before. To install URx, just do `pip3 install urx` and it will collect all the necessary dependencies and installed them.
3. It also uses Open3D for most of the computer vision tasks, so it is also needed to be installed. This package runs on ROS Noetic on Ubuntu 20.04, therefore, it should have Python 3.8.0. When I try to install it, I followed instructions from [Open3D](http://www.open3d.org/docs/release/getting_started.html). But I received a complaint `launchpadlib 1.10.13 requires testresources, which is not installed.` Then, I did `sudo apt install python3-testresources`. This helped and it does not complaint again. Tested `Open3D` is installed by invoking Python3 and import it. It should not complaint about no `Open3D` module.
4. 







