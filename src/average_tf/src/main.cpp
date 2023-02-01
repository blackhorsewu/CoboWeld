#include <iostream>
#include <chrono>
#include <thread>

#include <ros/ros.h>

#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>

#include <tf/tf.h>
#include <tf/transform_broadcaster.h>
#include <tf/transform_listener.h>

#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

class AverageTF
{
private:
  ros::NodeHandle nh;
  tf::TransformListener listener1, listener2, listener3, listener4;
  tf::TransformBroadcaster tr_br;
  tf::StampedTransform transform1, transform2, transform3, transform4; 

  // ros::Rate rate;

  std::thread tfListenerThread1, tfListenerThread2, tfListenerThread3, tfListenerThread4;

  std::string averagedFrame, averagedTargetFrame, sourceTargetFrame;
  std::string sourceFrame1, sourceFrame2, sourceFrame3, sourceFrame4;



  int averageSize;

  bool running;

  std::vector<double> poseAndOrientationVector1, poseAndOrientationVector2, poseAndOrientationVector3, poseAndOrientationVector4;   // x,y,z,rx,ry,rz,rw

  


public:
  AverageTF(ros::NodeHandle &n, char* args[]): nh(n), sourceTargetFrame(args[1]), sourceFrame1(args[2]), sourceFrame2(args[3]), sourceFrame3(args[4]), sourceFrame4(args[5]), averagedTargetFrame(args[6]), averagedFrame(args[7])
  {
    averageSize = std::atoi(args[8]);

    poseAndOrientationVector1.resize(7);
    poseAndOrientationVector2.resize(7);
    poseAndOrientationVector3.resize(7);
    poseAndOrientationVector4.resize(7);


    // listener1 = tf::TransformListener();
    // listener2 = tf::TransformListener();
    // listener3 = tf::TransformListener();
    // listener4 = tf::TransformListener();
  }

private:
  bool dataChecker(std::vector<double> transformVector)
  {
    if (std::isnan(transformVector[0]) || std::isnan(transformVector[1]) || std::isnan(transformVector[2]) ||
        std::isnan(transformVector[3]) || std::isnan(transformVector[4]) || std::isnan(transformVector[5]) ||
        std::isnan(transformVector[6]))
    {
      ROS_ERROR("some valuse are 'inf' or 'undefine'");
      return false;
    }
    else
      return true;
  }


  double average(std::vector<double> dataSet)
  {
    double sum = 0;
    for(size_t i = 0; i < dataSet.size(); ++i)
    {
      sum += dataSet[i];
    }
    return sum / static_cast<double> (dataSet.size());
  }


  void tfToVector(tf::StampedTransform &tr, std::vector<double> &vec)
  {
    vec[0] = tr.getOrigin().x();
    vec[1] = tr.getOrigin().y();
    vec[2] = tr.getOrigin().z();    
    vec[3] = tr.getRotation().x();
    vec[4] = tr.getRotation().y();
    vec[5] = tr.getRotation().z();
    vec[6] = tr.getRotation().w();
    // ROS_INFO_STREAM(vec[0] <<  vec[1] << vec[2] << vec[3] << vec[4] << vec[5] << vec[6]);
  }

  void tfListener(std::string targetFrame, std::string sourceFrame, int frameNum)
  {
    tf::TransformListener listener;
    tf::StampedTransform transform;
    std::vector<double> poseAndOrientationVector(7);    // x,y,z,rx,ry,rz,rw
    std::vector<double> x_buffer, y_buffer, z_buffer, rx_buffer, ry_buffer, rz_buffer, rw_buffer;
    double averaged_x, averaged_y, averaged_z, averaged_rx, averaged_ry, averaged_rz, averaged_rw;
    while (nh.ok())
    {
      try
      {
        listener.waitForTransform(targetFrame, sourceFrame, ros::Time(0), ros::Duration(5.0));
        listener.lookupTransform(targetFrame, sourceFrame, ros::Time(0), transform);
        
        tfToVector(transform, poseAndOrientationVector);
        // ROS_INFO_STREAM("poseAndOrietVec: " << ", " << poseAndOrientationVector[0] << ", " <<  poseAndOrientationVector[1] << ", " << poseAndOrientationVector[2] << ", " << poseAndOrientationVector[3] << ", " << poseAndOrientationVector[4] << ", " << poseAndOrientationVector[5] << ", " << poseAndOrientationVector[6]);
        
        if(dataChecker(poseAndOrientationVector))
        {
          if(x_buffer.size() == averageSize) 
          // if(x_buffer.size() && y_buffer.size() && z_buffer.size() && 
          //    rx_buffer.size() && ry_buffer.size() && rz_buffer.size() && rw_buffer.size() == averageSize)
          {
            averaged_x = average(x_buffer);
            averaged_y = average(y_buffer);
            averaged_z = average(z_buffer);
            averaged_rx = average(rx_buffer);
            averaged_ry = average(ry_buffer);
            averaged_rz = average(rz_buffer);
            averaged_rw = average(rw_buffer);

            switch (frameNum)
            {
              case 1:
                poseAndOrientationVector1[0] = averaged_x;
                poseAndOrientationVector1[1] = averaged_y;
                poseAndOrientationVector1[2] = averaged_z;
                poseAndOrientationVector1[3] = averaged_rx;
                poseAndOrientationVector1[4] = averaged_ry;
                poseAndOrientationVector1[5] = averaged_rz;
                poseAndOrientationVector1[6] = averaged_rw;
                break;
              case 2:
                poseAndOrientationVector2[0] = averaged_x;
                poseAndOrientationVector2[1] = averaged_y;
                poseAndOrientationVector2[2] = averaged_z;
                poseAndOrientationVector2[3] = averaged_rx;
                poseAndOrientationVector2[4] = averaged_ry;
                poseAndOrientationVector2[5] = averaged_rz;
                poseAndOrientationVector2[6] = averaged_rw;
                break;
              case 3:
                poseAndOrientationVector3[0] = averaged_x;
                poseAndOrientationVector3[1] = averaged_y;
                poseAndOrientationVector3[2] = averaged_z;
                poseAndOrientationVector3[3] = averaged_rx;
                poseAndOrientationVector3[4] = averaged_ry;
                poseAndOrientationVector3[5] = averaged_rz;
                poseAndOrientationVector3[6] = averaged_rw;
                break;
              case 4:
                poseAndOrientationVector4[0] = averaged_x;
                poseAndOrientationVector4[1] = averaged_y;
                poseAndOrientationVector4[2] = averaged_z;
                poseAndOrientationVector4[3] = averaged_rx;
                poseAndOrientationVector4[4] = averaged_ry;
                poseAndOrientationVector4[5] = averaged_rz;
                poseAndOrientationVector4[6] = averaged_rw;
                break;
            }

//            ROS_INFO("################################# UPDATED ######################################################");
            // ROS_INFO("tf updated!!!");      

            x_buffer.clear();
            y_buffer.clear();
            z_buffer.clear();
            rx_buffer.clear();
            ry_buffer.clear();
            rz_buffer.clear();
            rw_buffer.clear();
          }
          else
          {
            x_buffer.push_back(poseAndOrientationVector[0]);
            y_buffer.push_back(poseAndOrientationVector[1]);
            z_buffer.push_back(poseAndOrientationVector[2]);
            rx_buffer.push_back(poseAndOrientationVector[3]);
            ry_buffer.push_back(poseAndOrientationVector[4]);
            rz_buffer.push_back(poseAndOrientationVector[5]);
            rw_buffer.push_back(poseAndOrientationVector[6]);
          }
        }
      }
      catch (tf::TransformException &ex)
      {
        ROS_ERROR("%s",ex.what());
        ros::Duration(0.5).sleep();
        // continue;
      }
      ros::Rate(30).sleep();
    }
  }
  void stop()
  {
    tfListenerThread1.join();
    tfListenerThread2.join();
    tfListenerThread3.join();
    tfListenerThread4.join();
  }

public:
  void start()
  {
    tfListenerThread1 = std::thread(&AverageTF::tfListener, this, sourceTargetFrame, sourceFrame1, 1);
    tfListenerThread2 = std::thread(&AverageTF::tfListener, this, sourceTargetFrame, sourceFrame2, 2);
    tfListenerThread3 = std::thread(&AverageTF::tfListener, this, sourceTargetFrame, sourceFrame3, 3);
    tfListenerThread4 = std::thread(&AverageTF::tfListener, this, sourceTargetFrame, sourceFrame4, 4);

    double x, y, z, rx, ry, rz, rw;

    std::chrono::milliseconds duration(1000);
    std::this_thread::sleep_for(duration);

    while (nh.ok())
    {
      // ROS_INFO_STREAM("vec1: " << poseAndOrientationVector1[0] << ", " << poseAndOrientationVector1[1] << ", " << poseAndOrientationVector1[2] << ", " << poseAndOrientationVector1[3] << ", " << poseAndOrientationVector1[4] << ", " << poseAndOrientationVector1[5] << ", " << poseAndOrientationVector1[6]  );
      // ROS_INFO_STREAM("vec2: " << poseAndOrientationVector2[0] << ", " << poseAndOrientationVector2[1] << ", " << poseAndOrientationVector2[2] << ", " << poseAndOrientationVector2[3] << ", " << poseAndOrientationVector2[4] << ", " << poseAndOrientationVector2[5] << ", " << poseAndOrientationVector2[6]  );
      // ROS_INFO_STREAM("vec3: " << poseAndOrientationVector3[0] << ", " << poseAndOrientationVector3[1] << ", " << poseAndOrientationVector3[2] << ", " << poseAndOrientationVector3[3] << ", " << poseAndOrientationVector3[4] << ", " << poseAndOrientationVector3[5] << ", " << poseAndOrientationVector3[6]  );
      // ROS_INFO_STREAM("vec4: " << poseAndOrientationVector4[0] << ", " << poseAndOrientationVector4[1] << ", " << poseAndOrientationVector4[2] << ", " << poseAndOrientationVector4[3] << ", " << poseAndOrientationVector4[4] << ", " << poseAndOrientationVector4[5]<< ", "  << poseAndOrientationVector4[6]  );

      
      x = (poseAndOrientationVector1[0] + poseAndOrientationVector2[0] + poseAndOrientationVector3[0] + poseAndOrientationVector4[0]) / 4;
      y = (poseAndOrientationVector1[1] + poseAndOrientationVector2[1] + poseAndOrientationVector3[1] + poseAndOrientationVector4[1]) / 4;
      z = (poseAndOrientationVector1[2] + poseAndOrientationVector2[2] + poseAndOrientationVector3[2] + poseAndOrientationVector4[2]) / 4;
      rx = (poseAndOrientationVector1[3] + poseAndOrientationVector2[3] + poseAndOrientationVector3[3] + poseAndOrientationVector4[3]) / 4;
      ry = (poseAndOrientationVector1[4] + poseAndOrientationVector2[4] + poseAndOrientationVector3[4] + poseAndOrientationVector4[4]) / 4;
      rz = (poseAndOrientationVector1[5] + poseAndOrientationVector2[5] + poseAndOrientationVector3[5] + poseAndOrientationVector4[5]) / 4;
      rw = (poseAndOrientationVector1[6] + poseAndOrientationVector2[6] + poseAndOrientationVector3[6] + poseAndOrientationVector4[6]) / 4;

      tf::Vector3 t(x, y, z);
      tf::Quaternion q(rx, ry ,rz, rw);

      tf::Transform T(q.normalized(), t);

      if(std::isnan(x + y + z + rx + ry + rz +rw ))
      {
        continue;
      }
      else
      {
        try
        {
          tr_br.sendTransform(tf::StampedTransform(T, ros::Time::now(), averagedTargetFrame, averagedFrame));
        }
        catch (tf::TransformException& e)
        {
          ROS_ERROR ("transform exception: %s", e.what());
        }
        ros::Rate(15).sleep();
      }
    }
    stop();
  }
};


int main(int argc, char *argv[])
{
  ros::init(argc, argv, "averaged_tf");
  ros::NodeHandle n;


  AverageTF at(n, argv);
  at.start();

  return 0;
}
