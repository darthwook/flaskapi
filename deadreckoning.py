'''
Dead Reckoning using Trolley Tracker

'''

import pandas as pd
import math
import numpy as np
import matplotlib.pyplot as plt
import os
from numpy import sin,cos,pi
import lowpass as lp
import statistics
import computeSteps as cACC
import matplotlib.image as mpimg
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

DATA_PATH = 'C:/kafka_python/.venv/DeadReckoning/.venv/data/' #'./data/'
GRAPH_PATH = 'C:/kafka_python/.venv/DeadReckoning/.venv/graphs/'

#filter requirements
order = 4
fs = 10 #5000 #Frequency of Trolley Tracker set to 10Hz
cutoff = 2

# Function to create a session with retry strategy
def create_session():
    session = requests.Session()
    retries = Retry(
        total=5,  # Total number of retry attempts
        backoff_factor=1,  # Delay between retries 
        allowed_methods=["GET"],  # Retry only on GET requests
    )
    adapter = HTTPAdapter(max_retries=retries)
    session.mount("https://", adapter)
    return session

# Function to fetch data from Oracle APEX REST API
def fetch_db(url):
    """
    Fetch data from Oracle APEX REST API and return as a DataFrame.
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/110.0.0.0 Safari/537.36",
        "Accept": "application/json",  
    }

    session = create_session()

    try:
        response = session.get(url, headers=headers, timeout=10)
        response.raise_for_status()  # Raise HTTPError for bad responses
        json_data = response.json()
        if "items" in json_data:  # Ensure 'items' key exists
            return pd.DataFrame(json_data["items"])
        else:
            print(f"No 'items' key found in response: {json_data}")
            return pd.DataFrame()  # Return empty DataFrame if 'items' key is missing
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data from {url}: {e}")
        return pd.DataFrame()  # Return empty DataFrame on failure

# Oracle APEX REST API URLs
gyro_url = "https://apex.oracle.com/pls/apex/isha1/tables/gyro/?LIMIT=10"
accel_url = "https://apex.oracle.com/pls/apex/isha1/tables/accel/?limit=10"
mag_url = "https://apex.oracle.com/pls/apex/isha1/tables/magneto/?limit=10"

# Fetch data for gyro, accel, and magnetometer
gyro_data = fetch_db(gyro_url)
accel_data = fetch_db(accel_url)
mag_data = fetch_db(mag_url)

# Data processing function
def data_corrected(data):
    """
    Remove the mean from the data.
    """
    return np.array(data) - np.mean(data)

phone_acc = np.array([data_corrected(accel_data["accel_x"]),
                      data_corrected(accel_data["accel_y"]),
                      data_corrected(accel_data["accel_z"])])

phone_gyro = np.array([data_corrected(gyro_data["x"]),
                       data_corrected(gyro_data["y"]),
                       data_corrected(gyro_data["z"])])

if not mag_data.empty:
 phone_mag = np.array([data_corrected(mag_data["mag_x"]),
                      data_corrected(mag_data["mag_y"]),
                      data_corrected(mag_data["mag_z"])])


phone_acc_filtered = np.array([lp.butter_lowpass_filter(accel_data["accel_x"], cutoff, fs, order),
                                lp.butter_lowpass_filter(accel_data["accel_y"], cutoff, fs, order),
                                lp.butter_lowpass_filter(accel_data["accel_z"], cutoff, fs, order)])

phone_gyro_filtered = np.array([lp.butter_lowpass_filter(gyro_data["x"], cutoff, fs, order),
                                 lp.butter_lowpass_filter(gyro_data["y"], cutoff, fs, order),
                                 lp.butter_lowpass_filter(gyro_data["z"], cutoff, fs, order)])

phone_mag_filtered = np.array([lp.butter_lowpass_filter(mag_data["mag_x"], cutoff, fs, order),
                                lp.butter_lowpass_filter(mag_data["mag_y"], cutoff, fs, order),
                                lp.butter_lowpass_filter(mag_data["mag_z"], cutoff, fs, order)])

# Timestamp for Magnetometer
timestamp = mag_data["mag_time"]

pitch = gyro_data["x"]
roll  = gyro_data["y"]
yaw   = gyro_data["z"]

pitch_filtered = lp.butter_lowpass_filter(gyro_data["x"],cutoff,fs,order)
roll_filtered  = lp.butter_lowpass_filter(gyro_data["y"],cutoff,fs,order)
yaw_filtered   = lp.butter_lowpass_filter(gyro_data["z"],cutoff,fs,order)

# Rotation matrices
def R_x(x):
    # body frame rotation about x axis
    return np.array([[1,      0,       0],
                     [0,cos(-x),-sin(-x)],
                     [0,sin(-x), cos(-x)]])

def R_y(y):
    # body frame rotation about y axis
    return np.array([[cos(-y),0,-sin(-y)],
                    [0,      1,        0],
                    [sin(-y), 0, cos(-y)]])

def R_z(z):
    # body frame rotation about z axis
    return np.array([[cos(-z),-sin(-z),0],
                     [sin(-z), cos(-z),0],
                     [0,      0,       1]])
    
# Init arrays for new transformed accelerations
earth_mag = np.empty(phone_mag.shape)

for i in range(mag_data.shape[0]):
    earth_mag[:,i] = R_z(yaw[i]) @ R_y(roll[i]) @ R_x(pitch[i]) @ phone_mag[:,i]

x_data, y_data, z_data = earth_mag[0], earth_mag[1], earth_mag[2]

#timestamp = mag_data["mag_time"]

def compute_direction(x, y): #atan2 
    res = 0
    if y>0:
        res = 90 - math.atan(x/y)*180/math.pi
    elif y<0:
        res = 270 - math.atan(x/y)*180/math.pi
    else:
        if x<0:
            res = 180
        else:
            res = 0
    return res

def compute_compass(Hx, Hy):
    compass = []
    for i in range(len(Hx)):
        direction = compute_direction(Hx[i], Hy[i])
        compass.append(direction)
    return np.array(compass)

def compute_compass2(Hx, Hy):
    compass = []
    for i in range(len(Hx)):
        direction = compute_direction(Hx[i], Hy[i])
        direction = (450-direction)*math.pi/180
        direction = 2*math.pi - direction
        compass.append(direction)
    return np.array(compass)

def compute_compass3(Hx, Hy):
    compass = []
    for i in range(len(Hx)):
        direction = math.atan2(Hx[i],Hy[i])
        compass.append(direction)
    return np.array(compass)

def draw_arrows(x, y, angle, r=1, label=""):
    plt.arrow(x, y, r*math.sin(angle), r*math.cos(angle), head_width=.1, color="red",label=label)
    plt.arrow(x, y, 1, 0, head_width=.1, color="black")
    plt.annotate('N', xy=(x, y+r))
    plt.arrow(x, y, 0, 1, head_width=.1, color="black")
    plt.annotate('E', xy=(x+r, y))

def draw_all_arrows(data_x, data_y, data_angles, r=1):
    draw_arrows(data_x[0], data_y[0], data_angles[0], r, label="direction")
    for i in range(1,len(data_x)):
        draw_arrows(data_x[i], data_y[i], data_angles[i], r)


def compute_avg(data_x, data_y, steps):
    Hx_avg = []
    Hy_avg = []
    print('data x: ', data_x)
    print('steps: ', steps)
    
    if steps <= 0:
        raise ValueError("The number of steps must be greater than zero.")
    
    avg = int(len(data_x)/steps)
    x, y = 0, 0
    for i in range(len(data_x)):
        x += data_x[i]
        y += data_y[i]
        if (i+1)%avg == 0:
            Hx_avg.append(x/avg)
            Hy_avg.append(y/avg)
            x, y = 0, 0
    return np.array(Hx_avg), np.array(Hy_avg)

def make_graph_points():
    Hx = x_data
    Hy = y_data
    dataCompass = compute_compass(Hx,Hy)
    dataCompass2 = compute_compass2(Hx,Hy)
    #dataAcc = pull_data(DATA_PATH,'Accelerometer')[3] 
    dataAcc = fetch_db(accel_url)[3]

    print(dataAcc)
    timestamps = mag_data["time"]
    plt.plot(timestamps, dataAcc, marker='.', label=" steps")
    draw_all_arrows(timestamps, dataAcc, dataCompass2, 0.1)
    plt.title("")
    plt.xlabel("time [s]")
    plt.ylabel("acceleration norm [m/s^2]")
    plt.grid(True, which="both", linestyle="")
    plt.legend()
    plt.show()

def make_graph_steps(nbr_steps, distance_traveled):
    dist_step_avg = distance_traveled / nbr_steps
    dist_steps = []
    steps_num = np.arange(1,nbr_steps+1)
    i = 0
    for i in range(1,int(nbr_steps)+1):
        dist_steps.append(dist_step_avg*i)
        i += 1
    dist_steps = np.array(dist_steps)
    plt.plot(steps_num, dist_steps,"o",color="blue", label="steps")
    Hx = x_data
    Hy = y_data
    Hx, Hy = compute_avg(Hx,Hy,nbr_steps)
    dataCompass = compute_compass(Hx,Hy)
    dataCompass2 = compute_compass2(Hx,Hy)
    draw_all_arrows(steps_num, dist_steps, dataCompass2, 1)
    plt.title("Compas heading per step")
    plt.xlabel("Step number")
    plt.ylabel("Distance traveled [m]")
    plt.grid(True, which="both", linestyle="-")
    plt.legend()
    plt.show()

import matplotlib.pyplot as plt
import math
import base64
from io import BytesIO

def draw_arrows_target(x, y, angle, r=1, label="", display_direction=0):
    plt.arrow(x, y, dx=r * math.sin(angle), dy=r * math.cos(angle), head_width=.3, color="red", label=label)
    if display_direction == 1:
        plt.arrow(x, y, 1, 0, head_width=.1, color="black")
        plt.annotate('N', xy=(x, y + r))
        plt.arrow(x, y, 0, 1, head_width=.1, color="black")
        plt.annotate('E', xy=(x + r, y))
    return (x + r * math.sin(angle), y + r * math.cos(angle))

def draw_all_arrows_target(x0, y0, data_angles, r=1.5, display_steps=1, display_direction=0, display_stepsNumber=0):
    x, y = draw_arrows_target(x0, y0, data_angles[0], r, label="direction", display_direction=display_direction)
    plt.plot(x0, y0, ".", color="blue", label="steps")
    plt.annotate('{}'.format(1), xy=(x0 + 0.3, y0), color="blue")

   # coordinates = []
   # coordinates.append({'timestamp': time.time(), 'x': x0, 'y': y0, 'trolley_id': '1'})  # STATIC trolley ID
    
    for i in range(1, len(data_angles)):
        if display_steps == 1:
            plt.plot(x, y, ".", color="blue")
        if display_stepsNumber == 1:
            plt.annotate('{}'.format(i + 1), xy=(x + 0.3, y), color="blue")
        
        # Store coordinates for each step
       # coordinates.append({'timestamp': time.time(), 'x': x, 'y': y, 'trolley_id': '1'}) # STATIC trolley ID
        
        x, y = draw_arrows_target(x, y, data_angles[i], r, display_direction=display_direction)

    #save_coord(coordinates)



def make_target(nbr_steps, distance_traveled, display_steps=1, display_direction=0, display_stepsNumber=0):
    Hx = earth_mag[0]
    Hy = earth_mag[1]
    Hx, Hy = compute_avg(Hx, Hy, nbr_steps)
    dataCompass = compute_compass(Hx, Hy)
    dataCompass2 = compute_compass2(Hx, Hy)

    plt.figure()  # Start a new figure
    draw_all_arrows_target(0, 0, dataCompass2, 1.5, display_steps, display_direction, display_stepsNumber)
    plt.legend()

    # Save the figure to a BytesIO object
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()  # Close the figure to free memory

    # Convert the image to a base64 string
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    return f'data:image/png;base64,{img_str}'  # Return the base64 string

import time
import json 

# Set the APEX REST API URL
POST_URL = "https://apex.oracle.com/pls/apex/isha1/tables/trolleys_log/"  

def save_coord(steps_data, trolley_id):

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/110.0.0.0 Safari/537.36",
        "Accept": "application/json",
        "Content-Type": "application/json",
    }

    coordinates = []
    
    for i, data in enumerate(steps_data):
        x = data['x']
        y = data['y']
        timestamp = data['timestamp']
        steps = int(np.max(cACC.compute(display_graph=0, without_mean=1)))
        
        coordinates.append({
            "x": x,
            "y": y,
            "timestamp": timestamp,
            "trolley_id": trolley_id,
            "steps": steps
        })

    payload = {
        "coordinates": coordinates
    }

    print('Payload:', payload)

    # Send the data as a JSON POST request to the API
    try:
        response = requests.post(POST_URL, json=payload, headers=headers)
        
        print(f"Payload Sent: {json.dumps(payload, indent=2)}")
        print(f"Response Status Code: {response.status_code}")
        print(f"Response Text: {response.text}")
        
        if response.status_code == 200:
            print("All steps saved successfully.")
        else:
            print("Failed to save steps.")
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")


def calculate_steps_and_save(data_angles, x0=0, y0=0, r=1.5, trolley_id="1"): #STATIC TROLLEY_ID
    
    coordinates = []
    coordinates.append({'timestamp': time.time(), 'x': x0, 'y': y0, 'trolley_id': trolley_id}) 
    
    # Calculate coordinates for each step and record them
    x, y = x0, y0
    for i in range(len(data_angles)):
        x, y = calculate_new_coordinates(x, y, data_angles[i], r)
        
        # Store coordinates for each step
        coordinates.append({'timestamp': time.time(), 'x': x, 'y': y, 'trolley_id': trolley_id})
    
    # Call save_coord function to send coordinates to the API
    save_coord(coordinates, trolley_id)

def calculate_new_coordinates(x, y, angle, r=1):
    return (x + r * math.sin(angle), y + r * math.cos(angle))

def make_target_with_image(nbr_steps, distance_traveled, display_steps=1, display_direction=0, display_stepsNumber=0):
    Hx = earth_mag[0]
    Hy = earth_mag[1]
    Hx, Hy = compute_avg(Hx, Hy, nbr_steps)
    dataCompass = compute_compass(Hx, Hy)
    dataCompass2 = compute_compass2(Hx, Hy)
    calculate_steps_and_save(dataCompass2, x0=0, y0=0, r=1.5, trolley_id="1") #STATIC TROLLEY_ID
    
    '''
    # Draw the arrows and plot the graph over the image
    draw_all_arrows_target(0, 0, dataCompass2, 1.5, display_steps, display_direction, display_stepsNumber)
    
    plt.legend()
    plt.savefig(GRAPH_PATH + 'tracking_with_image')
    plt.show()
    '''


if __name__ == "__main__":
    
    steps = np.max(cACC.compute(display_graph = 0, without_mean=1))
    #  steps = np.max(cACC.compute(display_graph =0)
    dist = steps*0.69
    
    make_target_with_image(steps,
                dist,
                display_steps = 1,
                display_direction = 0,
                display_stepsNumber = 0)
    
