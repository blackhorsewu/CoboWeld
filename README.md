# CoboWeld


# <p align="center"> Chinese National Engineering Research Centre for Steel Structure (Hong Kong Branch) </p>
# <p align="center"> 国家钢结构结构工程技术研究中心（香港分中心）</p>
1 February 2023. (Friday)

# General Introduction
This is a CNERC project. The purpose of this project is to use Cooperative Robots for Steel Structures in Constructions.

# Implementation details
Structure of the system
CoboWeld is implemented using Universal Robots UR5 robotic arm as the primary action provider.

The whole system software is implemented on the ROS platform. It is ROS Noetic on Ubuntu 20.04. The software is a mixture in C++ and Python 3.8.

Visual images are provided by Intel Realsense D435 RGB-D camera.

Most of the Computer Vision functions are developed using Open3D in Python 3.8

## Installation of the Universal Robots ROS Driver
The UR5 robotic arm is driven by ROS via the Universal Robots ROS Driver. It is provided via [this](https://github.com/UniversalRobots/Universal_Robots_ROS_Driver) repository in Github.

# Visualisation via RViz of ROS - a Digital Twin of the system
ROS provides a system visualisation tool, RViz. Rviz takes a description of the robot, from now on it will be called a manipulator. This description is in a URDF (Universal Robot Description Format) file.

Universal Robots provided this description file. A copy of [this](src/universal_robot/ur5_moveit_config) file is in this depository.

CoboWeld itself has a sub-package, coboweld_support, that provides all the supporting facilities the system needs. In coboweld_support, there is a urdf folder. There is an integrated urdf file. Integrated in this coboweld435.xacro file are the working environment, that is the world, the welding table, and the tools. The tools are, the welding torch and the sensor, that is the RGB-D camera - Intel Realsense D435.

# The MoveIt package of ROS
On the ROS platform there is a package, MoveIt, that helps to plan and move the manipulator UR5, in both the virtual world in RViz and in the real world.

Before it can be used it must be installed and configured. It also provides a `moveit_setup_assistant` to do the setup for it.

Invoke the `setup assistant`:
```
roslaunch moveit_setup_assistant setup_assistant.launch
```

Once it is invoked, click `Create New MoveIt Configuration Package`. It is also necessary to load a URDF:
```
~/CoboWeld/src/coboweld_support/urdf/coboweld435.xacro
```
After the URDF is loaded:
1. it will be in the `Self-Collisions` tab, click `Generate Collision Matrix`
2. `Virtual Joints` tab, click `Add Virtual Joint`. Virtual Joint Name should be set to `FixedBase`, Parent Frame Name set to `world`. Then push the `Save` button
3. `Planning Groups` tab, click `Add Group`. Group Name `manipulator`. Kinematic Solver `ur_kinematics/UR5KinematicsPlugin`. Click the `Add Kin. Chain` button, then click `Expand All` button at the bottom. Choose `base_link` and click the `Choose Selected` button for the `Base Link` field. Then choose `torch` and click the `Choose Selected` button for the `Tip Link` field. Finally, click `Save` button.
4. `Robot Poses` tab, Click `Add Pose` button. Fill in the `Pose Name` field with `up`, then move the robot all the way up by:
 * `shoulder_pan_joint`: 1.5708
 * `shoulder_lift_joint`: -1.5708
 * `wrist_1_joint`: -1.5708   
push the `Save` button.
5. Add one more `Robot Poses` and call it `home`:
6. Go to `Author Information` tab and fill it in.
7. Configuration Files tab. For the `Save Path` put down `src/coboweld435_moveit_config`
8. Press the `Generate Package` button. It will then complaint `No end effectors have been added`, just press `OK`. It will then display `Configuration package generated successfully!`
9. Exit Setup Assistant.

Afterwards, it will generate a package, in the case of coboweld, coboweld_moveit_config. 

---

This is a project of the CNERC
## Chinese National Engineering Research Centre for Steel Structure (Hong Kong Branch)


It is based on previous work by Jeffery Zhou and Maggie.
That was almost completely in Python.

It is going to be partly ROS C++ and mostly in python.
It also heavily depends on Open3D.

It is going to use the URx (Python) for actually driving
the robotic arm UR5 (maybe UR10 later) instead of using
MoveIt of ROS.

This project started on this 19 July, 2022. (Tuesday).

### Find orientation of Waypoints on the Welding Path

---

After I have upgraded from melodic to noetic, it seems 
I need to do some more for the Intel Realsense. The 
details are [here](https://answers.ros.org/question/363889/intel-realsens-on-ubuntu-2004-ros-noetic-installation-desription/).   

I will have to do this anyway if I want to use Realsense, 
especially if I want to do hand eye calibration. 

---

# Installation of this package

RobotWeld depends on:
1. C++, Python3, UR Robot Driver
2. It uses URx to control the UR5 or R10. Therefore, it is necessary to install URx. It is now much simpler than before. To install URx, just do `pip3 install urx` and it will collect all the necessary dependencies and installed them.
3. Currently, it is now using a special version, that I have modified and stored in `coboweld_core/src/script/urx/urrobot.py`. This is necessary because the `digital_out` will only be switch on while a program is running. If for any reason, the e-switch on the teach pendent is pushed, it should switch `off` the welding machine, otherwise it would be very dangerous. However, URx sends every instruction to UR5 as a program. Therefore, once the `set_digital_out` instruction is sent, there is no program running and hence the `digital_out` will be switched off. To overcome this problem, I have modified URx, in specific, the `urrobot.py`. A function called `execute_ls` added. That will take an added parameter, `output`, to switch `on` before moving the welding torch along the welding path and switch `off` after the path is traversed. (3 April, 2023.)
4. It also uses Open3D for most of the computer vision tasks, so it is also needed to be installed. This package runs on ROS Noetic on Ubuntu 20.04, therefore, it should have Python 3.8.0. When I tried to install it, I followed instructions from [Open3D](http://www.open3d.org/docs/release/getting_started.html). But I received a complaint `launchpadlib 1.10.13 requires testresources, which is not installed.` Then, I did `sudo apt install python3-testresources`. This helped and it does not complaint again. Tested `Open3D` is installed by invoking Python3 and import it. It should not complaint about no `Open3D` module.

---

# Universal Robots ROS Driver

When try to launch the `setup455.launch`, it complaint that it is always waiting ro the `scaled_vel_joint_traj_controller` and cannot connect to the `Action client`, with the following errors:
```
Waiting for /scaled_vel_joint_traj_controller/follow_joint_trajectory to come up

Action client not connected: /scaled_vel_joint_traj_controller/follow_joint_trajectory
```

The current driver uses a `scaled_vel_joint_traj_controller`. However, the `ur_robot_driver/launch/ur5_bringup.launch` file, one of the arguments `name="controllers"`, default=, only include the `scaled_pos_joint_traj_controller`. One of the ways to enable the velocity controller is to just change the `pos` to `vel`.

---

# Realsense D455 (6 February 2023)

When launching the system, with the Realsense D455 camera, it always has the following error message:
```
(messenger-libusb.cpp:42) control_transfer returned error, index: 768, error: Resource temporarily unavailable, number: 11
```

When searching with Google, there seems to be quite a lot of people have the same problem. Then a `StefanT83` replied in the [github issues](https://github.com/IntelRealSense/realsense-ros/issues/2386), with the following solution:
```
Hi, I got the same type of error indicated in the title, with my D455 on Ubuntu 20.04, and eventually managed to find a solution, so I decided to share my story here. My feeling is that there needs to be a match between the firmware of the D455 (I used 5.12.15.50) and librealsense2 (I used 2.51) and the ROS wrapper.
Step1. removed any previous install related to 'realsense':
$ dpkg -l | grep "realsense" | cut -d " " -f 3 | xargs sudo dpkg --purge
Step2. installed librealsense2 according to https://github.com/IntelRealSense/librealsense/blob/master/doc/distribution_linux.md > Installing the packages
Make sure to unplug the sensor at the end, wait 10 seconds, then plug it back again. Test
$realsense-viewer
and saw the depth camera is operational.
Step3. Check version of librealsense that was just installed:
$dpkg -l|grep realsense
where I got 2.51
Step4. Firmware upgrade/downgrade of the D455 to match the librealsense 2.51
Now my interpretation is that we need to check the column 'SDK ver' on https://dev.intelrealsense.com/docs/firmware-releases and match that to, in my case, librealsense 2.51
Save the bin file and flash it to the D455 using https://www.intelrealsense.com/developers/ > Firmware update guide > 'Firmware Update Tool (rs-fw-update)'
Make sure to unplug the sensor, wait 10 seconds, then plug it back again.
Step5. Install RealSense Wrapper https://github.com/IntelRealSense/realsense-ros > Method 2
Step6. Test in ROS:
$roslaunch realsense2_camera rs_camera.launch
Troubleshooting: in case of errors, unplug the sensor, wait 10 seconds, then plug it back again.

I tested this approach on a D435 and was successful too. It worked on Ubuntu 20.04 native as well as a virtual image on VMWare Player 16.
Cheers!
``` 


# open3d_ros_helper
7 February 2023.

Both Open3d and ROS are used in CoboWeld, the two platforms use different formats for their Point Clouds. Therefore, there is a need to convert Point Clouds between the 2 systems. There is now a package `open3d_ros_helper` that provides conversion routines and works with Open3d, ROS Noetic and Python 3. 

It is necessary to install a package `ros-noetic-ros-numpy` first. Then install other dependencies as follow:
```
pip3 install numpy==1.20 open3d opencv-python pyrsistent
pip3 install open3d_ros_helper
```

Then comes the crucial part. When we look at the github for the [open_3d_helper](https://github.com/SeungBack/open3d-ros-helper), we find there is a commit remark `cloud_array ravel added to rospc_to_o3dpc`. `rospc_to_o3dpc` is exactly the function that is needed to convert ROS pointcloud2 to Open3d point clouds. After getting into the folder of `open3d_ros_helper`, there is the `open3d_ros_helper.py` file with a commit remark next to this file name. When this remark is clicked, it will show exactly where has been changed on the date `1 April 2021`. On line 261 of this file, `.ravel()` was added to the end.

It is absolutely **important** to follow this and add this `.ravel()` to the end of line 261 in the file `~/.local/lib/python3.8/site-packages/open3d_ros_helper/open3d_ros_helper.py`. Otherwise, when the program is run, a RuntimeError will complaint that `o3d.colors = open3d.utility.Vector3dVector(rgb_npy)`.




