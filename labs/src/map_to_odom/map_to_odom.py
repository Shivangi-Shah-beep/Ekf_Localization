#!/usr/bin/env python

import rospy
from geometry_msgs.msg import PoseWithCovarianceStamped, TransformStamped, Transform, Pose
import tf2_ros
import tf_conversions
from tf_conversions import posemath
import math

class MapOdomPublisherNode:
    def __init__(self):
        # Create a TransformBroadcaster
        self.br = tf2_ros.TransformBroadcaster()

        # Create a TF Buffer and Listener to get transforms
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        # Subscriber to the EKF pose
        rospy.Subscriber('/ekf/pose', PoseWithCovarianceStamped, self.ekf_pose_callback)

        # Parameters
        self.publish_rate = 30.0  # Hz

        # Internal state
        self.map_to_odom = None  # TransformStamped object

        # Start a timer to publish the transform at 30Hz
        self.timer = rospy.Timer(rospy.Duration(1.0 / self.publish_rate), self.timer_callback)

        rospy.loginfo("Map-Odom Publisher Node Initialized")

    def transform_to_pose(self, transform):
        rospy.loginfo('Transforming now')
        pose = Pose()
        pose.position.x = transform.translation.x
        pose.position.y = transform.translation.y
        pose.position.z = transform.translation.z
        pose.orientation = transform.rotation
        return pose

    def ekf_pose_callback(self, msg):
        # This function is called when a new pose is received from the EKF Localization node
        rospy.loginfo('Recived pose message')
        try:
            # Get the odom->base_link transform
            self.tf_buffer.can_transform('odom', 'base_link', rospy.Time(0), rospy.Duration(1.0))
            trans = self.tf_buffer.lookup_transform('odom', 'base_link', rospy.Time(0), rospy.Duration(1.0))
        except (tf2_ros.LookupException, tf2_ros.ExtrapolationException, tf2_ros.ConnectivityException) as e:
            rospy.logwarn('Could not get transform from odom to base_link: %s', e)
            return

        # Get the pose of base_link in the map frame from the EKF
        map_to_base_link_pose = msg.pose.pose  # geometry_msgs/Pose

        # Convert odom->base_link Transform to Pose
        odom_to_base_link_pose = self.transform_to_pose(trans.transform)

        # Convert Poses to KDL Frames
        odom_to_base_link_frame = posemath.fromMsg(odom_to_base_link_pose)
        map_to_base_link_frame = posemath.fromMsg(map_to_base_link_pose)

        # Invert odom->base_link to get base_link->odom
        base_link_to_odom_frame = odom_to_base_link_frame.Inverse()

        # Compute map->odom Frame
        map_to_odom_frame = map_to_base_link_frame * base_link_to_odom_frame

        # Convert map->odom Frame to TransformStamped
        map_to_odom = TransformStamped()
        map_to_odom.header.stamp = rospy.Time.now()
        map_to_odom.header.frame_id = 'map'
        map_to_odom.child_frame_id = 'odom'

        # Convert KDL Frame to Transform
        transform = Transform()
        transform.translation.x = map_to_odom_frame.p[0]
        transform.translation.y = map_to_odom_frame.p[1]
        transform.translation.z = map_to_odom_frame.p[2]

        (x, y, z, w) = map_to_odom_frame.M.GetQuaternion()
        transform.rotation.x = x
        transform.rotation.y = y
        transform.rotation.z = z
        transform.rotation.w = w

        map_to_odom.transform = transform

        # Update internal state
        self.map_to_odom = map_to_odom

        rospy.loginfo("Updated map->odom transform based on new EKF pose.")

    def timer_callback(self, event):
        #rospy.loginfo("Timer callback")
        # This function is called at 30Hz to publish the transform
        if self.map_to_odom is not None:
            # Update the timestamp
            self.map_to_odom.header.stamp = rospy.Time.now()
            self.br.sendTransform(self.map_to_odom)
def main():
   rospy.init_node('map_odom_publisher_node', anonymous=True)
   rospy.loginfo('Starting map to odom node')
   node = MapOdomPublisherNode()
   rospy.spin()

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass








