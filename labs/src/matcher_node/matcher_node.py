#!/usr/bin/env python

import numpy
import rospy
import message_filters
import threading
from opencv_apps.msg import Point2DArrayStamped
from opencv_apps.msg import Point2D
from std_msgs.msg import Float64
from std_msgs.msg import Float64MultiArray
from std_msgs.msg import MultiArrayLayout
from std_msgs.msg import MultiArrayDimension

class Correspondence:
    def __init__(self, color, pad_rejected= False, standalone = False):
        self.lock = threading.Lock()
        self.color = color
        self.pad_rejected = pad_rejected
        if standalone:
            z_meas = message_filters.Subscriber(f'goodfeature_{color}/corners', Point2DArrayStamped)
            z_pred = message_filters.Subscriber(f'predicted_{color}/points', Point2DArrayStamped)
        
        self.pub_delta_z = rospy.Publisher(f'matched_{color}/delta_z', Float64MultiArray, queue_size=10)

        if standalone:
            self.synced_sub = message_filters.ApproximateTimeSynchronizer([ z_meas, z_pred ], queue_size = 4, slop = 0.1)
            self.synced_sub.registerCallback(self.z_callback)

    def minimun_distance(self, pred_point, meas_points):
        best_point = None
        best_distance = float('inf')
        for meas_point in meas_points:
            distance = numpy.linalg.norm([ numpy.array(pred_point) - numpy.array(meas_point) ]
)
            if distance < best_distance:
                best_distance = distance
                best_point = meas_point
        return best_point, best_distance

    def z_callback(self, meas, pred):
        with self.lock:
            predictions = [ (pred.x, pred.y) for pred in pred.points ]
            measurements = [ (meas.x, meas.y) for meas in meas.points ]
            self.match_correspondences(measurements, predictions)

    def match_correspondences(self, observed_points, predicted_points):
        actual_matches, expected_matches, distance_errors = [], [],[]
        
        for expected in predicted_points:
            closest_point, min_dist = self.minimun_distance(expected, observed_points)

            if closest_point is None:
                if self.pad_rejected:
                    actual_matches.extend([expected[0], expected[1]])
                else:
                    continue
            else:
                actual_matches.extend([closest_point[0], closest_point[1]])

            expected_matches.extend([expected[0], expected[1]])
            distance_errors.append(min_dist)

        z_error = [obs - exp for obs, exp in zip(actual_matches, expected_matches)]
        rospy.loginfo(f"Residual Error (Delta z) for {self.color} = {z_error}")

        return actual_matches, expected_matches, z_error


def main():
    rospy.init_node('correspondence_matcher')
    colors = ['red', 'green', 'yellow', 'cyan', 'magenta']
    pad_rejected = False

    matchers = []  # Store matchers if needed later

    for color in colors:
        rospy.loginfo(f'Starting correspondence_matcher for {color}')
        matcher = Correspondence(color, pad_rejected, standalone=True)
        matchers.append(matcher)  # Keep track if needed

    rospy.spin()
    rospy.loginfo('Done')


if __name__=='__main__':
    main()
