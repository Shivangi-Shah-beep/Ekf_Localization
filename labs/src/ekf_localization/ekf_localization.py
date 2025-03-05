#!/usr/bin/env python

import rospy
import numpy
import math
import json
import threading

from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseWithCovarianceStamped
from opencv_apps.msg import Point2DArrayStamped
from measurement_model_2 import measurement_model_2
from matcher_node import matcher_node

class Measurement:
    def __init__(self, x, y, height, radius,
                 landmark_color, tcx_bias, tcy_bias, tcz_bias, covariance):
        self.predictor = measurement_model_2.FeaturePredictor(
            x, y,height, radius, landmark_color,
            tcx_bias, tcy_bias, tcz_bias
        )
        self.matcher = matcher_node.Correspondence(landmark_color)
        self.measurement_cov = covariance

    def get_jacobian(self, x, y, theta):
        return self.predictor.get_jacobian(x, y, theta)

    def measurement(self, x, y, theta, measured_points):
        predicted_points = self.predictor.predict_points(x, y, theta)
        if predicted_points is None:
            return None
        matched_measurements, matched_predictions, delta_z = self.matcher.match_correspondences(measured_points, predicted_points)
        
        cov = numpy.identity(8) * self.measurement_cov

        return matched_measurements, matched_predictions, delta_z, cov

class EKFLocalization:
    def __init__(self, map_filename, x, y, theta,
                 hl, rl, tcx_bias, tcy_bias, tcz_bias,
                 default_pix_cov, default_pos_cov,
                 alpha1, alpha2, alpha3, alpha4):
        self.lock = threading.Lock()
        self.ready = False

        with open(map_filename, 'r') as f:
            self.map_dict = json.load(f)
            rospy.loginfo('Map loaded')
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.alpha3 = alpha3
        self.alpha4 = alpha4
        self.measurement_models = {}
        self.measurement_subscribers = {}
        self.set_state(x, y, theta, default_pos_cov)
        self.vel_lin = None
        self.vel_ang = None
        self.time = None

        self.pub_pose = rospy.Publisher('/ekf/pose', PoseWithCovarianceStamped, queue_size = 1)

        for landmark_color, coordinates in self.map_dict.items():
            model = Measurement(
                coordinates['x'], coordinates['y'], hl, rl, landmark_color, 
                tcx_bias, tcy_bias, tcz_bias, default_pix_cov
            )
            self.measurement_models[landmark_color] = model  

            topic_name = f'/goodfeature_{landmark_color}/corners'
            subscriber = rospy.Subscriber(topic_name, Point2DArrayStamped, 
                                        lambda msg, col=landmark_color: self.measurement_callback(msg, col))
            self.measurement_subscribers[landmark_color] = subscriber

        self.odom_sub = rospy.Subscriber('/odometry/filtered', Odometry, self.odom_callback, queue_size = 1)
        self.ready = True

    def odom_callback(self, msg):
        #rospy.loginfo("Received odometry message")
        with self.lock:
            if not self.ready:
                return
            
            self.vel_lin = msg.twist.twist.linear.x
            self.vel_ang = msg.twist.twist.angular.z

            if self.time is None:
                self.time = msg.header.stamp.to_sec()
                return
            
            delta_t = msg.header.stamp.to_sec() - self.time

            self.x, self.y, theta_pred, cov = self.motion_update(delta_t)
            if numpy.linalg.det(self.cov) < 3:
                self.cov += cov
            self.theta = math.atan2(math.sin(theta_pred), math.cos(theta_pred))
            self.time = msg.header.stamp.to_sec()
            self.publish_pose()
    
    def measurement_callback(self, m, color):
        #rospy.loginfo("The measurements have been recieved")
        with self.lock:
            if not self.ready:
                rospy.logwarn('The node is not ready')
                return
            
            if len(m.points) < 4:
                rospy.logwarn(f'Not enough points for {color}')
                return
            
            if self.vel_lin is None or self.vel_ang is None:
                rospy.logwarn('Odometry not recieved')
                return
            
            if self.time is None:
                self.time = m.header.stamp.to_sec()
                return
            
            delta_t = m.header.stamp.to_sec() - self.time
            if delta_t < -0.05:
                rospy.logwarn(f'stale measurement dropped for {color} at {delta_t}')          
                return
            if delta_t < 0:
                delta_t = 0
            measurement_updated = self.measurement_update(m, color, delta_t)
            self.time += delta_t
            self.publish_pose()
    
    def motion_update(self, delta_t):
        st = math.sin(self.theta)
        ct = math.cos(self.theta)
        omega_delta_t = delta_t * self.vel_ang
        if abs(self.vel_ang) > 1e-5:
            codt = math.cos(self.theta + omega_delta_t)
            sodt = math.sin(self.theta + omega_delta_t)
            r = self.vel_lin / self.vel_ang
            G = numpy.array([
                [ 1, 0, -r*ct + r*codt ],
                [ 0, 1, -r*st + r*sodt ],
                [ 0, 0, delta_t ]
            ])
            V = numpy.array([
                [ (-st + sodt)/self.vel_ang, r*(st - sodt)/self.vel_ang + r*codt*delta_t ],
                [ (ct - codt)/self.vel_ang, -r*(ct - codt)/self.vel_ang + r*sodt*delta_t ],
                [ 0, delta_t ],
            ])
            delta_mu = numpy.array([
                [ -r*st + r*sodt ],
                [ r*ct - r*codt ],
                [ omega_delta_t ]
            ])
        else:
            v_delta_t = delta_t * self.vel_lin
            G = numpy.array([
                [ 1, 0, -v_delta_t*st ],
                [ 0, 1, v_delta_t*ct ],
                [ 0, 0, delta_t ]
            ])
            V = numpy.array([
                [ delta_t * ct, -v_delta_t*delta_t*st/2 ],
                [ delta_t * st, v_delta_t*delta_t*ct/2 ],
                [ 0, delta_t ]
            ])
            delta_mu = numpy.array([
                [ v_delta_t * ct ],
                [ -v_delta_t * st ],
                [ omega_delta_t ]
            ])
        omega_sq = self.vel_ang * self.vel_ang
        v_sq = self.vel_lin * self.vel_lin
        M = numpy.array([
            [ self.alpha1 * v_sq + self.alpha2 * omega_sq, 0 ],
            [ 0, self.alpha3 * v_sq + self.alpha4 * omega_sq ]
        ])
        mu = numpy.array([
            [ self.x ],
            [ self.y ],
            [ self.theta ]
        ]) + delta_mu
        x,y,theta= mu[0][0], mu[1][0], mu[2][0]
        cov = G @ self.cov @ G.transpose() + V @ M @ V.transpose()
        return x , y, theta, cov

    def measurement_update(self, m, color, delta_t):
        if delta_t > 0:
            x_pred, y_pred, theta_pred, cov_pred = self.motion_update(delta_t)
            self.cov = cov_pred
        else:
            x_pred = self.x
            y_pred = self.y
            theta_pred = self.theta
        
        measurement_vector = [(point.x, point.y) for point in m.points[:4]]

        measurement_result = self.measurement_models[color].measurement(x_pred, y_pred, theta_pred, measurement_vector)
        
        if measurement_result is None:
            self.x = x_pred
            self.y = y_pred
            self.theta = math.atan2(math.sin(theta_pred), math.cos(theta_pred))
            return False
        matched_meas, matched_pred, delta_z, measurement_cov = measurement_result

        H = self.measurement_models[color].get_jacobian(
            x_pred, y_pred, theta_pred)
        
        S = H @ self.cov @ H.T + measurement_cov
        K = self.cov @ H.T @ numpy.linalg.inv(S)
        mu_pred = numpy.array([
            [ x_pred ],
            [ y_pred ],
            [ theta_pred ]
        ])
        mu = mu_pred + K @ numpy.array([ delta_z ]).T
        Sigma = (numpy.identity(3) - K @ H) @ self.cov
        self.x = mu[0,0]
        self.y = mu[1,0]
        self.theta = math.atan2(math.sin(mu[2,0]), math.cos(mu[2,0]))
        self.cov = Sigma
        rospy.loginfo(f'X={self.x},y={self.y}, theta={self.theta}, covariance= {self.cov}')
        return True


    def publish_pose(self):
        pose = PoseWithCovarianceStamped()
        pose.header.stamp = rospy.Time(self.time)
        pose.header.frame_id = 'map'
        pose.pose.covariance = [
            self.cov[0,0], self.cov[0,1], 0, 0, 0, self.cov[0, 2],
            self.cov[1,0], self.cov[1,1], 0, 0, 0, self.cov[1, 2],
            0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, self.cov[2,2]
        ]
        pose.pose.pose.position.x = self.x
        pose.pose.pose.position.y = self.y
        pose.pose.pose.position.z = 0.0
        pose.pose.pose.orientation.x = 0.0
        pose.pose.pose.orientation.y = 0.0
        pose.pose.pose.orientation.z = math.sin(self.theta/2)
        pose.pose.pose.orientation.w = math.cos(self.theta/2)
        self.pub_pose.publish(pose)
        rospy.loginfo('pose published')

    def set_state(self, x, y, theta, cov):
        with self.lock:
            self.x = x
            self.y = y
            self.theta = theta
            self.cov = numpy.array([[ cov, 0, 0],
                                    [ 0, cov, 0],
                                    [ 0 , 0, cov]])

    

def main():
    rospy.init_node('ekf_localization_2', anonymous=True)
    rospy.loginfo('starting ekf_localization')
    alpha1, alpha2, alpha3, alpha4 = 0.02, 0.01, 0.01, 0.02
    map_file = "/home/shivangi_shah/catkin_ws/src/prob_rob_labs/labs/maps/landmark_map.json"
    tcx_bias = 0.0 
    tcy_bias = 0.0 
    tcz_bias = 0.060 
    start_x, start_y, start_theta = 0, 0, 0
    hl, rl = 0.5, 0.1 
    default_pix_cov, default_pos_cov = 3, 0.4
    ekf_localization = EKFLocalization(map_file, start_x, start_y, start_theta,
                              hl, rl, tcx_bias, tcy_bias, tcz_bias,
                              default_pix_cov, default_pos_cov,
                              alpha1, alpha2, alpha3, alpha4)
    rospy.spin()
    rospy.loginfo('done')

if __name__=='__main__':
    main()