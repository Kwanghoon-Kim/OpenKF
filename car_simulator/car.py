# copyrights © Mohanad Youssef 2023

import numpy as np
from sensor import Sensor
import time, threading
import matplotlib.pyplot as plt

class Pose(object):
    def __init__(self):
        self.x = 0.
        self.y = 0.
        self.yaw = 0.

class Car(object):
    def __init__(self, ax, width_m, wheelbase_m, track_m, front_to_axle_m, rear_to_axle_m):
        # create a lock
        self.lock = threading.Lock()

        self.length_m = wheelbase_m + front_to_axle_m + rear_to_axle_m
        self.width_m = width_m
        self.wheelbase_m = wheelbase_m
        self.track_m = track_m
        self.rear_to_axle_m = rear_to_axle_m
        self.front_to_axle_m = front_to_axle_m

        self.contour_pts = np.zeros((5, 2)) # four points describing the vehicle contour
        self.steering_arrow = np.zeros((2, 2)) # two points
        self.build_contour()

        self.max_forward_speed = 5.5
        self.max_reverse_speed = 3.5
        self.max_steer_angle = np.deg2rad(35.)
        
        self.pose = Pose()
        self.velocity = 0.
        self.steering_angle = 0.
        self.yaw_rate = 0.
        self.acceleration = 0.
        self.steering_rate = 0.

        self.wheel_l_velocity_mps = 0.
        self.wheel_r_velocity_mps = 0.

        self.max_buff_len = 10000
        self.actual_x_lst = []
        self.actual_y_lst = []
        self.meas_x_lst = []
        self.meas_y_lst = []
        self.meas_wheel_l_velocity_lst = []
        self.meas_wheel_r_velocity_lst = []
        self.meas_gyro_lst = []

        self.prev_time = round(time.time()*1000)

        gps_cov = np.eye(2) * 1.0
        self.sens_gps = Sensor(self.gps_model, gps_cov)

        wheel_cov = np.eye(2) * 0.05**2
        self.sens_wheel = Sensor(self.wheel_model, wheel_cov)

        gyro_cov = np.eye(1) * np.deg2rad(1)**2
        self.sens_gyro = Sensor(self.gyro_model, gyro_cov)

        self.update_states()
        self.update_sens_gps(ax[0])
        self.update_sens_wheel(ax[1])
        self.update_sens_gyro(ax[2])

        self.threads_running = True

    def build_contour(self):
        self.contour_pts[0, :] = np.array([-self.rear_to_axle_m, self.width_m / 2.0]) # top left
        self.contour_pts[1, :] = np.array([self.wheelbase_m + self.front_to_axle_m, self.width_m / 2.0])
        self.contour_pts[2, :] = np.array([self.wheelbase_m + self.front_to_axle_m, -self.width_m / 2.0])
        self.contour_pts[3, :] = np.array([-self.rear_to_axle_m, -self.width_m / 2.0])
        self.contour_pts[4, :] = self.contour_pts[0, :]
        # print(f'self.contour_pts=\n{self.contour_pts}\n')

        self.steering_arrow[0, :] = np.array([0.0, 0.0]) # fixed point
        self.steering_arrow[1, :] = np.array([1.5, 0.0]) # rotating point with steering angle


    def plot(self, ax):
        rot_matrix = np.array(
            [[np.cos(self.pose.yaw), -np.sin(self.pose.yaw)],
             [np.sin(self.pose.yaw), np.cos(self.pose.yaw)]])
        
        translate_vec = np.array([self.pose.x, self.pose.y]).T
        
        tf_pts = np.zeros((5, 2)) 
        tf_pts[0, :] = rot_matrix @ self.contour_pts[0, :].T + translate_vec
        tf_pts[1, :] = rot_matrix @ self.contour_pts[1, :].T + translate_vec
        tf_pts[2, :] = rot_matrix @ self.contour_pts[2, :].T + translate_vec
        tf_pts[3, :] = rot_matrix @ self.contour_pts[3, :].T + translate_vec
        tf_pts[4, :] = tf_pts[0, :]
        ax.plot(tf_pts[:, 0], tf_pts[:, 1], c='blue')


        steer_rot_matrix = np.array(
            [[np.cos(self.steering_angle), -np.sin(self.steering_angle)],
             [np.sin(self.steering_angle), np.cos(self.steering_angle)]])

        tf_steering_pts = np.zeros((2, 2))
        steering_offset_dist = np.array([self.wheelbase_m, 0.0])
        tf_steering_pts[0, :] = self.steering_arrow[0, :] + steering_offset_dist
        tf_steering_pts[1, :] = steer_rot_matrix @ self.steering_arrow[1, :] + steering_offset_dist

        tf_steering_pts[0, :] = rot_matrix @ tf_steering_pts[0, :] + translate_vec
        tf_steering_pts[1, :] = rot_matrix @ tf_steering_pts[1, :] + translate_vec
        ax.plot(tf_steering_pts[:, 0], tf_steering_pts[:, 1], c='green')

        self.lock.acquire()
        # trajectory
        ax.plot(self.actual_x_lst, self.actual_y_lst)

        # gps sensor measurements
        ax.scatter(self.meas_x_lst, self.meas_y_lst, s=2, c='red', alpha=0.5)
        self.lock.release()


    def set_acceleration(self, acceleration):
        self.acceleration = acceleration
        

    def set_steering_rate(self, steering_rate):
        self.steering_rate = steering_rate


    def update(self, dt):
        self.velocity += self.acceleration * dt
        self.steering_angle += self.steering_rate * dt
        
        self.steering_angle = min(self.steering_angle, self.max_steer_angle)
        self.steering_angle = max(self.steering_angle, -self.max_steer_angle)
        
        self.velocity = min(self.velocity, self.max_forward_speed)
        self.velocity = max(self.velocity, -self.max_reverse_speed)
        # print(f'velocity=\n{self.velocity}\n')
        
        # bicycle kinematic model
        self.yaw_rate = self.velocity * np.tan(self.steering_angle) / self.wheelbase_m
        
        # accumulating position and angle
        self.pose.yaw += self.yaw_rate * dt
        self.pose.x += self.velocity * np.cos(self.pose.yaw) * dt
        self.pose.y += self.velocity * np.sin(self.pose.yaw) * dt

        self.lock.acquire()
        # store path
        self.actual_x_lst.append(self.pose.x)
        self.actual_y_lst.append(self.pose.y)

        # prevent huge elements in the lists
        if len(self.actual_x_lst) > self.max_buff_len:
            self.actual_x_lst.pop(0)
        if len(self.actual_y_lst) > self.max_buff_len:
            self.actual_y_lst.pop(0)
        self.lock.release()

        A = np.array([[1., 1.], [1., -1.]])
        b = np.array([[2*self.velocity], [self.track_m * self.yaw_rate]])
        x = np.linalg.lstsq(A, b)[0] # x = [right wheel, left wheel]

        self.wheel_r_velocity_mps, self.wheel_l_velocity_mps = x[0], x[1]


    def brake(self):
        self.velocity = 0.
        self.acceleration = 0.


    def stop_threads(self):
        self.car_thread.cancel()
        self.gps_thread.cancel()
        self.wheel_thread.cancel()
        self.gyro_thread.cancel()


    def run_threads(self):
        self.update_states()
        self.update_sens_gps()        
        self.update_sens_wheel()
        self.update_sens_gyro()
    

    def gps_model(self, x):
        return x


    def wheel_model(self, x):
        return x


    def gyro_model(self, x):
        return x


    def update_states(self):

        curr_time = round(time.time()*1000)
        dt_s = (curr_time - self.prev_time) / 1000
        # print(f"update_states : dt = {dt_s} sec")
        self.prev_time = curr_time

        self.update(dt=dt_s)

        # configure a timer thread
        time_in_seconds = 0.01
        self.car_thread = threading.Timer(time_in_seconds, self.update_states)
        self.car_thread.start()


    def update_sens_gps(self, ax):
        # curr_time = round(time.time()*1000)
        # print(curr_time)
        self.sens_gps.generate(np.array([self.pose.x, self.pose.y]))

        self.lock.acquire()
        self.meas_x_lst.append(self.sens_gps.meas[0])
        self.meas_y_lst.append(self.sens_gps.meas[1])
        if len(self.meas_x_lst) > self.max_buff_len:
            self.meas_x_lst.pop(0)
        if len(self.meas_y_lst) > self.max_buff_len:
            self.meas_y_lst.pop(0)
        self.lock.release()

        # configure a timer thread
        time_in_seconds = 0.1
        self.gps_thread = threading.Timer(time_in_seconds, self.update_sens_gps, args={ax})
        self.gps_thread.start()


    def update_sens_wheel(self, ax):
        self.lock.acquire()
        actual = np.array([self.wheel_l_velocity_mps, self.wheel_r_velocity_mps])
        actual = actual.reshape((2,))
        self.sens_wheel.generate(actual)

        if (abs(self.sens_wheel.meas[0]) < 0.2):
            self.sens_wheel.meas[0] = 0.

        if (abs(self.sens_wheel.meas[1]) < 0.2):
            self.sens_wheel.meas[1] = 0.

        self.meas_wheel_l_velocity_lst.append(self.sens_wheel.meas[0])
        self.meas_wheel_r_velocity_lst.append(self.sens_wheel.meas[1])

        if len(self.meas_wheel_l_velocity_lst) > self.max_buff_len:
            self.meas_wheel_l_velocity_lst.pop(0)
        if len(self.meas_wheel_r_velocity_lst) > self.max_buff_len:
            self.meas_wheel_r_velocity_lst.pop(0)

        self.lock.release()

        time_in_seconds = 0.02
        self.wheel_thread = threading.Timer(time_in_seconds, self.update_sens_wheel, args={ax})
        self.wheel_thread.start()

    def update_sens_gyro(self, ax):
        self.lock.acquire()

        actual_yaw_rate = np.array([self.yaw_rate])
        self.sens_gyro.generate(actual_yaw_rate)

        self.meas_gyro_lst.append(self.sens_gyro.meas[0])

        if len(self.meas_gyro_lst) > self.max_buff_len:
            self.meas_gyro_lst.pop(0)

        self.lock.release()

        time_in_seconds = 0.02
        self.gyro_thread = threading.Timer(time_in_seconds, self.update_sens_gyro, args={ax})
        self.gyro_thread.start()