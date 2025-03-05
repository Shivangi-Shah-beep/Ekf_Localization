import rospy
import tf2_ros
import numpy as np
import tf.transformations as tf
import threading
import math
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import CameraInfo
from opencv_apps.msg import Point2D, Point2DArrayStamped
from measurement_model import measurement_model

class FeaturePredictor:
    def __init__(self, xl, yl, hl, rl, color,
                 tcx_bias, tcy_bias, tcz_bias, standalone = False):
        

        self.color = color
        self.tx = None
        self.ty = None
        self.tz = None
        self.fx = None
        self.fy = None
        self.cx = None
        self.cy = None
        self.xl = xl
        self.yl = yl
        self.hl = hl
        self.rl = rl
        self.image_height = None
        self.image_width = None
        self.image_frame = None
        self.tcx_bias = tcx_bias
        self.tcy_bias = tcy_bias
        self.tcz_bias = tcz_bias

        self.lock = threading.Lock()

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.z_model, self.z_jacobian = measurement_model.measurement_model()
        rospy.loginfo(f'measurement model ready for {color}')
        
        self.cc_sub = rospy.Subscriber('/front/left/camera_info', CameraInfo, self.camera_callback, queue_size = 10)

        if standalone:
            self.gt_sub = rospy.Subscriber('/jackal/ground_truth/pose', PoseStamped, self.gt_callback, queue_size = 10)
            
        self.expected_points_pub = rospy.Publisher('predicted_{}/points'.format(color), Point2DArrayStamped, queue_size = 10)
       
    def camera_callback(self, msg):
        #rospy.loginfo("Finding camera caliberations")
        with self.lock:
            self.fx = msg.K[0]
            self.fy = msg.K[4]
            self.cx = msg.K[2]
            self.cy = msg.K[5]
            self.image_height = msg.height
            self.image_width = msg.width
            self.image_frame = msg.header.frame_id
            self.get_transforms

    def get_transforms(self):
        #rospy.loginfo('Getting the transforms')
        if self.tx is None:
            t = self.tf_buffer.lookup_transform('base_link', 'front_camera', rospy.Time())
            self.tx = t.transform.translation.x + self.tcx_bias
            self.ty = t.transform.translation.y + self.tcy_bias
            self.tz = t.transform.translation.z + self.tcz_bias
            rospy.loginfo(f'tx: {self.tx}, ty= {self.ty},tz={self.ty}')
        return True

    def gt_callback(self, p):
        with self.lock:
            x = p.pose.position.x
            y = p.pose.position.y
            _, _, theta = tf.euler_from_quaternion([
                p.pose.orientation.x,
                p.pose.orientation.y,
                p.pose.orientation.z,
                p.pose.orientation.w
            ])

            predicted_points = self.predict_points(x, y, theta)
            if not predicted_points:
                rospy.logwarn("No predicted points")
                return

            # Call the new function to publish points
            self.publish_predicted_points(predicted_points, p.header.stamp)
            

    def publish_predicted_points(self, points, timestamp):
        """ Publishes predicted points in a separate function """
        points_msg = Point2DArrayStamped()
        points_msg.header.stamp = timestamp
        points_msg.header.frame_id = self.image_frame

        # Convert predicted points to message format
        points_msg.points = [Point2D(x=p[0], y=p[1]) for p in points]

        # Publish the message
        self.expected_points_pub.publish(points_msg)
        rospy.loginfo(f"Published {len(points)} predicted points")



    def predict_points(self, x, y, theta):
        if not self.get_transforms():
            rospy.logwarn('Tranforms not recieved')
            return None
        
        if self.fx is None:
            rospy.logwarn('Camera Parameters not recieved')
            return None

        predicted_points=self.z_model(
            x, y, theta, self.tx, self.ty, self.tz,
            self.xl, self.yl, self.hl, self.rl,
            self.fx, self.fy, self.cx, self.cy)
        
        predicted_points= self.filter_valid_points(predicted_points)

        if len(predicted_points) < 4:
            return None
        
        return predicted_points

    def get_jacobian(self, x, y, theta):
        if not self.get_transforms():
            return None
        if self.fx is None:
            return None
        return self.z_jacobian(x, y, theta,
                               self.tx, self.ty, self.tz,
                               self.xl, self.yl, self.hl, self.rl,
                               self.fx, self.fy, self.cx, self.cy)
    
    def filter_valid_points(self, points):
        filtered_points = [
            (round(points[i][0]), round(points[i + 1][0])) 
            for i in range(0, len(points), 2)
            if 0 <= points[i][0] < self.image_width and 0 <= points[i + 1][0] < self.image_height
        ]
        return filtered_points

def main():
    rospy.init_node('measurement_predictor')
    rospy.loginfo("Node initialized")

    landmarks = {
        'red': (8.5, -5),
        'green': (8.5, 5),
        'yellow': (-11.5, 5),
        'magenta': (-11.5, -5),
        'cyan': (0, 0)
    }

    tcx_bias = 0.0
    tcy_bias = 0.0
    tcz_bias = 0.060

    hl = 0.5  # Landmark height
    rl = 0.1 # Landmark radius

    # Dictionary to store predictor instances for all landmarks
    predictors = {}

    for color, (xl, yl) in landmarks.items():
        rospy.loginfo(f"Initializing FeaturePredictor for {color}")
        predictors[color] = FeaturePredictor(xl, yl, hl, rl, color, tcx_bias, tcy_bias, tcz_bias)

    rospy.spin()


if __name__ == '__main__':
    main()