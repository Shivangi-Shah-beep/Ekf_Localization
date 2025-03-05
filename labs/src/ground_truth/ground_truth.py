#!/usr/bin/env python

import rospy
from gazebo_msgs.msg import LinkStates
from geometry_msgs.msg import PoseStamped, TwistStamped


class GroundTruthNode:
    def __init__(self):
        
        rospy.Subscriber('/gazebo/link_states', LinkStates, self.link_states_callback, queue_size=10)
        # Set up publishers for PoseStamped and TwistStamped messages
        self.pose_pub = rospy.Publisher('/jackal/ground_truth/pose', PoseStamped, queue_size=10)
        self.twist_pub = rospy.Publisher('/jackal/ground_truth/twist', TwistStamped, queue_size=10)

        # Store the name of the link we're interested in (jackal::base_link)
        self.link_name = 'jackal::base_link' 

        rospy.loginfo("Ground Truth Node has started...")

    def link_states_callback(self, msg):
        # Extract the index of the base_link frame
        rospy.loginfo("Twist and pose data being published")

        try:
            index = msg.name.index(self.link_name)
        except ValueError:
            rospy.logwarn("Link %s not found in /gazebo/link_states", self.link_name)
            return

        # Extract the pose and twist (velocity) for the base_link
        pose = msg.pose[index]
        twist = msg.twist[index]

        # Create PoseStamped message
        pose_stamped = PoseStamped()
        pose_stamped.header.stamp = rospy.Time.now()
        pose_stamped.header.frame_id = "map"
        pose_stamped.pose = pose

        # Create TwistStamped message
        twist_stamped = TwistStamped()
        twist_stamped.header.stamp = rospy.Time.now()
        twist_stamped.header.frame_id = "base_link"
        twist_stamped.twist = twist

        # Publish the extracted pose and twist
        self.pose_pub.publish(pose_stamped)
        self.twist_pub.publish(twist_stamped)

        rospy.loginfo(pose)
        rospy.loginfo(twist)

        # Log the linear.x value and timestamp to file    

    
def main():
    rospy.init_node('ground_truth_node', anonymous= True)
    node = GroundTruthNode()
    rospy.spin()

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass