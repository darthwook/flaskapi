"""
computing the number of steps from data provided by accelerometer
"""

import math
import matplotlib.pyplot as plt
import numpy as np
import lowpass as lp
import peakAccelThreshold as pat
import jk_threshold as asjt
from scipy.signal import find_peaks
import os
import peakDetection as pd
import statistics
import deadreckoning as dr

DATA_PATH = 'C:/kafka_python/.venv/DeadReckoning/.venv/data/'
GRAPH_PATH = './graphs/'


def pull_data(dir_name, file_name):
    f = open(dir_name + '/' + file_name + '.csv')
    xs = []
    ys = []
    zs = []
    rs = []
    timestamps = []
    line_counter = 0
    for line in f:
        if line_counter > 0:
            value = line.split(',')
            if len(value) > 3:
                timestamps.append(float(value[1]))  # 'seconds' column
                z = float(value[2])                 # 'z' column
                y = float(value[3])                 # 'y' column
                x = float(value[4])                 # 'x' column
                r = math.sqrt(x ** 2 + y ** 2 + z ** 2)
                '''
                timestamps.append(float(value[-4]))
                x = float(value[-3])
                y = float(value[-2])
                z = float(value[-1])
                r = math.sqrt(x ** 2 + y ** 2 + z ** 2)
                '''
                xs.append(x)
                ys.append(y)
                zs.append(z)
                rs.append(r)
        line_counter += 1
    return np.array(xs), np.array(ys), np.array(zs), np.array(rs), np.array(timestamps)

def data_without_mean(data):
    return np.array(data) - statistics.mean(data)

def lowpass_filter(data, cutoff, fs, order):
    return lp.butter_lowpass_filter(data, cutoff, fs, order)


def peak_accel_threshold(data, timestamps, cst):
    return pat.peak_accel_threshold(data, timestamps, cst)


def compute__peak_accel_threshold(data, timestamps, cst, step_length=0.69):

    crossings = pat.peak_accel_threshold(data, timestamps, cst)
    steps = len(crossings)//2
    length = step_length
    distance_traveled = steps * length
    return steps, distance_traveled


def graph__peak_accel_threshold(data, timestamps, cst, step_length=0.69):

    crossings = pat.peak_accel_threshold(data, timestamps, cst)
    print('crossing: ', crossings)
    steps = len(crossings)//2
    print('steps: ', steps)
    length = step_length
    distance_traveled = steps * length
    print('dist: ', distance_traveled)
    plt.title("Peak Acceleration Threshold: {} steps, {} m".format(steps,round(distance_traveled,2)))
    plt.xlabel("Time [sec]")
    plt.ylabel("Acceleration Norm [m/s^2]")
    plt.grid()
    plt.plot(timestamps,data,'b-', linewidth=2)
    plt.plot(timestamps, np.full(shape=len(timestamps), fill_value=cst, dtype=float),'r',linewidth=0.5)
    plt.plot(crossings.T[0], crossings.T[1], 'ro', linewidth=0.5)
    plt.savefig(GRAPH_PATH+'compute_steps_method1')
    plt.show() 


def compute__jk_threshold(data, timestamps, cst, step_length=0.69):
  
    jumps, avg = asjt.adaptive_step_jk_threshold(data, timestamps, cst)
    steps = len(jumps)
    distance = step_length * steps
    return steps, distance, jumps, avg


def graph__jk_threshold(data, timestamps, cst):

    steps, distance, jumps, avg = compute__jk_threshold(data, timestamps, cst)
    ts = [jump['ts'] for jump in jumps]
    val = [jump['val'] for jump in jumps]
    plt.title("Adaptive Step Jerk Threshold: {} steps, {} m".format(steps, round(distance,2)))
    plt.xlabel("Time [sec]")
    plt.ylabel("Acceleration Norm [m/s^2]")
    plt.grid()
    plt.plot(timestamps, data, 'b-', linewidth=2)
    plt.plot(ts, val, 'ro')
    plt.savefig(GRAPH_PATH+'compute_steps_method2')
    plt.show()


def compute__find_peaks(data, timestamps, distance=60, prominence=0.5, step_length=0.69):

    peaks, properties = find_peaks(data, distance=distance, prominence=prominence)
    steps = len(peaks)
    distance_traveled = steps * step_length
    return steps, distance_traveled, peaks, properties 


def graph__find_peaks(data, timestamps, distance=60, prominence=0.5, step_length=0.69):
 
    steps, distance_traveled, peaks, properties = compute__find_peaks(data,timestamps,distance,prominence,step_length)
    plt.title("find_peaks method: {} steps, {} m".format(steps, round(distance_traveled,2)))
    plt.xlabel("Time [sec]")
    plt.ylabel("Acceleration Norm [m/s^2]")
    plt.grid()
    plt.plot(timestamps, data, 'b-', linewidth=2)
    plt.plot(timestamps[peaks],data[peaks],'x', color="red", label="peaks detected")
    plt.legend()
    plt.savefig(GRAPH_PATH+'compute_steps_method3')
    plt.show()


def compute__peakdetect(data, timestamps, lookahead=1, delta=3/4, step_length=0.69):
 
    _max, _min = pd.peakdetect(data,timestamps, lookahead, delta)
    xm = [p[0] for p in _max]
    ym = [p[1] for p in _max]
    steps = len(_max)
    distance = steps * step_length
    return steps, distance, xm, ym
    
    
def graph__peakdetect(data, timestamps, lookahead=1, delta=3/4, step_length=0.69):
  
    steps, distance, xm, ym = compute__peakdetect(data,timestamps,lookahead, delta, step_length)
    plt.title("peakdetect method: {} steps, {} m".format(steps, round(distance,2)))
    plt.xlabel("Time [sec]")
    plt.ylabel("Acceleration Norm [m/s^2]")
    plt.grid()
    plt.plot(timestamps, data, 'b-', linewidth=2)
    plt.plot(xm,ym,'x', color="red", label="peaks detected")
    plt.legend()
    plt.savefig(GRAPH_PATH+'compute_steps_method4')
    plt.show()


def compute(display_graph=1, without_mean=0):
    
    GRAVITY = 9.81
    
    #filter requirements
    order = 4
    fs = 10 #100 
    cutoff = 2
    accel_url = "https://apex.oracle.com/pls/apex/isha1/tables/accel/"


    #x_data, y_data, z_data, r_data, timestamps = pull_data(DATA_PATH, 'Accelerometer')

    accel_data = dr.fetch_db(accel_url)
    x_data = accel_data["accel_x"].to_numpy()  
    y_data = accel_data["accel_y"].to_numpy()
    z_data = accel_data["accel_z"].to_numpy()
    timestamps = accel_data["seconds_elapsed"].to_numpy()
    r_data = np.sqrt(x_data ** 2 + y_data ** 2 + z_data ** 2) 
    print('time: ', timestamps)
   
    #filter
    r = lowpass_filter(r_data, cutoff, fs, order)
    print('r: ', r)

    #mean
    if without_mean == 1:
        r = data_without_mean(r) #removing mean from data
        ZERO = 0
    else:
        ZERO = GRAVITY
        
    step1, dist1 = compute__peak_accel_threshold(r,timestamps,ZERO) 
    if display_graph == 1: graph__peak_accel_threshold(r,timestamps,ZERO)
    
    return step1 



if __name__ == "__main__":

 
 compute(display_graph=1, without_mean=0)
           # without_mean=0)
  


