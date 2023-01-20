# copyrights © Mohanad Youssef 2023

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from car import Car, Pose
from sensor import Sensor
import keyboard
from pynput import mouse

# initial data
data = []

# create figure and axes objects
fig1, ax1 = plt.subplots()  # top view
fig2, ax2 = plt.subplots(2) # wheels and gyro sensors inputs

ax = (ax1, ax2[0], ax2[1])

car = Car(ax, 2.7, 2.91, 1.55, 0.8, 0.6)

scene_lim_delta = 20

def on_scroll(x, y, dx, dy):
    global scene_lim_delta
    scene_lim_delta -= dy

def plot(i):
    ax1.clear()
    ax1.grid()
    ax1.set_xlim([car.pose.x - scene_lim_delta, car.pose.x + scene_lim_delta])
    ax1.set_ylim([car.pose.y - scene_lim_delta, car.pose.y + scene_lim_delta])
    car.plot(ax1)

def print_states():
    ax1.text(0.01, 0.95, f'acceleration: {round(car.acceleration, 2)} m/s2', horizontalalignment='left', verticalalignment='center', transform=ax1.transAxes)
    ax1.text(0.01, 0.9, f'velocity: {round(car.velocity, 2)} m/s', horizontalalignment='left', verticalalignment='center', transform=ax1.transAxes)
    ax1.text(0.01, 0.85, f'steering angle: {round(np.rad2deg(car.steering_angle), 2)}°', horizontalalignment='left', verticalalignment='center', transform=ax1.transAxes)

def animate1(i):
    plot(i)
    print_states()

def animate2(i):
    ax2[0].clear()
    ax2[0].grid()
    ax2[1].clear()
    ax2[1].grid()

    car.lock.acquire()
    x1 = np.linspace(0, len(car.meas_wheel_l_velocity_lst), len(car.meas_wheel_l_velocity_lst))
    wheel_l_data, = ax2[0].plot(x1, car.meas_wheel_l_velocity_lst, c='red')
    wheel_r_data, = ax2[0].plot(x1, car.meas_wheel_r_velocity_lst, c='blue')
    wheel_l_data.set_label('left wheel velocity (m/s)')
    wheel_r_data.set_label('right wheel velocity (m/s)')

    x2 = np.linspace(0, len(car.meas_gyro_lst), len(car.meas_gyro_lst))
    gyro_data, = ax2[1].plot(x2, car.meas_gyro_lst, c='red')
    gyro_data.set_label('gyro yaw rate (rad/s)')

    car.lock.release()

    ax2[0].legend()
    ax2[1].legend()

def main():
    keyboard.on_press_key('up arrow', lambda _: car.set_acceleration(2.5))
    keyboard.on_press_key('down arrow', lambda _: car.set_acceleration(-2.5))
    keyboard.on_press_key('left arrow', lambda _: car.set_steering_rate(np.deg2rad(30.0)))
    keyboard.on_press_key('right arrow', lambda _: car.set_steering_rate(-np.deg2rad(30.0)))
    keyboard.on_press_key(' ', lambda _: car.brake())
    
    keyboard.on_release_key('up arrow', lambda _: car.set_acceleration(0.0))
    keyboard.on_release_key('down arrow', lambda _: car.set_acceleration(0.0))
    keyboard.on_release_key('left arrow', lambda _: car.set_steering_rate(np.deg2rad(0.0)))
    keyboard.on_release_key('right arrow', lambda _: car.set_steering_rate(np.deg2rad(0.0)))
    
    keyboard.on_press_key('r', lambda _: car.run_threads())
    keyboard.on_press_key('t', lambda _: car.stop_threads())

    # ...or, in a non-blocking fashion:
    listener = mouse.Listener(on_scroll=on_scroll)
    listener.start()

    # call the animation every 100ms
    ani1 = FuncAnimation(fig1, animate1, interval=20)
    ani2 = FuncAnimation(fig2, animate2, interval=20)
    plt.show()

    car.stop_threads()

if __name__ == '__main__':
    main()