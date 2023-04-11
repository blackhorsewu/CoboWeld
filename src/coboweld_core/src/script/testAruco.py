import roslib
import sys
import rospy
import cv2
from std_msgs.msg import String
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped
from cv_bridge import CvBridge, CvBridgeError
import time
import numpy as np
from scipy.spatial.transform import Rotation as R

bridge = CvBridge()

ARUCO_DICT = {
	"DICT_4X4_50": cv2.aruco.DICT_4X4_50,
	"DICT_4X4_100": cv2.aruco.DICT_4X4_100,
	"DICT_4X4_250": cv2.aruco.DICT_4X4_250,
	"DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
	"DICT_5X5_50": cv2.aruco.DICT_5X5_50,
	"DICT_5X5_100": cv2.aruco.DICT_5X5_100,
	"DICT_5X5_250": cv2.aruco.DICT_5X5_250,
	"DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
	"DICT_6X6_50": cv2.aruco.DICT_6X6_50,
	"DICT_6X6_100": cv2.aruco.DICT_6X6_100,
	"DICT_6X6_250": cv2.aruco.DICT_6X6_250,
	"DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
	"DICT_7X7_50": cv2.aruco.DICT_7X7_50,
	"DICT_7X7_100": cv2.aruco.DICT_7X7_100,
	"DICT_7X7_250": cv2.aruco.DICT_7X7_250,
	"DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
	"DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
	"DICT_APRILTAG_16h5": cv2.aruco.DICT_APRILTAG_16h5,
	"DICT_APRILTAG_25h9": cv2.aruco.DICT_APRILTAG_25h9,
	"DICT_APRILTAG_36h10": cv2.aruco.DICT_APRILTAG_36h10,
	"DICT_APRILTAG_36h11": cv2.aruco.DICT_APRILTAG_36h11
}

aruco_type = "DICT_5X5_100"

arucoDict = cv2.aruco.Dictionary_get(ARUCO_DICT[aruco_type])

arucoParams = cv2.aruco.DetectorParameters_create()

# All the zeros must be 0.0 not just 0; they must be float
intrinsic_camera = np.array(
    ((921.0864868164062, 0.0, 647.822021484375),
     (0.0, 920.7874755859375, 352.6297302246094),
     (0.0, 0.0 , 1.0)
    )
  )
distortion = np.array((0.0, 0.0, 0.0, 0.0, 0.0))

def pose_estimation(frame, aruco_dict_type, matrix_coefficients, distortion_coefficients):

  aruco_pose = PoseStamped()
  gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  cv2.aruco_dict = cv2.aruco.Dictionary_get(aruco_dict_type)
  parameters = cv2.aruco.DetectorParameters_create()


  corners, ids, rejected_img_points = cv2.aruco.detectMarkers(
      gray,
      cv2.aruco_dict,
      parameters=parameters
    )
    #cameraMatrix=matrix_coefficients,
    #distCoeff=distortion_coefficients)

      
  if len(corners) > 0:
    for i in range(0, len(ids)):
        
      rvec, tvec, markerPoints = cv2.aruco.estimatePoseSingleMarkers(
          corners[i], 
          0.0245, 
          matrix_coefficients,
          distortion_coefficients
        )
      tvec = tvec[0,0]
      #print('tvec: ', tvec.shape)
      rvec = rvec[0,0]
      #print('rvec: ', rvec.shape)
      r = R.from_rotvec(rvec)
      orientation = r.as_quat()
      # It should be possible to construct a pose to be published in RViz
      # A pose consists of (a) position x, y, z, (b) orientation in quaternion x, y, z, w
      # position
      aruco_pose.pose.position.x = tvec[0]
      aruco_pose.pose.position.y = tvec[1]
      aruco_pose.pose.position.z = tvec[2]
      # orientation
      aruco_pose.pose.orientation.x = orientation[0]
      aruco_pose.pose.orientation.y = orientation[1]
      aruco_pose.pose.orientation.z = orientation[2]
      aruco_pose.pose.orientation.w = orientation[3]

      # pose.header.frame_id = 'd435_color_optical_frame'
      aruco_pose.header.frame_id = 'd435_color_optical_frame'
      aruco_pose.header.stamp = rospy.Time.now()
      pub_pose.publish(aruco_pose)

      cv2.aruco.drawDetectedMarkers(frame, corners) 

      #cv2.aruco.drawAxis(frame, matrix_coefficients, distortion_coefficients, rvec, tvec, 0.01)
      cv2.drawFrameAxes(frame, matrix_coefficients, distortion_coefficients, rvec, tvec, 0.01)  

  return frame

def callback(data):
  # print('I am here 4.\n')
  cv_image = bridge.imgmsg_to_cv2(data, "bgr8")
  # cv2.imshow('Image window', cv_image)
  output = pose_estimation(cv_image, ARUCO_DICT[aruco_type], intrinsic_camera, distortion)
  # print(output)
  # cv2.imshow('Estimated Pose', output)

  cv2.waitKey(0)

# print('I am here 1.')
rospy.init_node('testAruco', anonymous=True)

pub_pose = rospy.Publisher('/ArUCo', PoseStamped, queue_size=1)

# print('I am here 2.')

# rospy.Subscriber('/d435/color/image_raw', Image, callback, queue_size=1)

# print('I am here 3.\n')

''''''
while not rospy.is_shutdown():
  data = rospy.wait_for_message('/d435/color/image_raw', Image, timeout=10)
  cv_image = bridge.imgmsg_to_cv2(data, "bgr8")
  pose_estimation(cv_image, ARUCO_DICT[aruco_type], intrinsic_camera, distortion)
  #time.sleep(1)

cv2.destroyAllWindows()