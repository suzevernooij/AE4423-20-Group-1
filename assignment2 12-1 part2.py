# -*- coding: utf-8 -*-
"""
Created on Sun Jan 12 10:32:13 2025

@author: julia
"""

# Import necessary libraries
  
import numpy as np 
import math  
import copy  
import pandas as pd  
import matplotlib.pyplot as plt 
from sklearn.linear_model import LinearRegression
from scipy.spatial import distance_matrix
import networkx as nx
from collections import defaultdict

#import data Suze
# airport_data = r"C:\Users\suzev\OneDrive\Documents\Master\AE4423-20 Airline Planning and Optimisation\Assignment 2\AE4423Ass2\AirportData.xlsx"
# fleet_data = r"C:\Users\suzev\OneDrive\Documents\Master\AE4423-20 Airline Planning and Optimisation\Assignment 2\AE4423Ass2\FleetType.xlsx"
# demand_data = r"C:\Users\suzev\OneDrive\Documents\Master\AE4423-20 Airline Planning and Optimisation\Assignment 2\AE4423Ass2\Group1.xlsx"

#import data Maaike




#import data Julia
airport_data = r"C:\TIL\Jaar 2\Airline Planning\Opdracht2\AE4423Ass2\AirportData.xlsx"
fleet_data = r"C:\TIL\Jaar 2\Airline Planning\Opdracht2\AE4423Ass2\FleetType.xlsx"
demand_data = r"C:\TIL\Jaar 2\Airline Planning\Opdracht2\AE4423Ass2\Group1.xlsx"


# Read Excel Files
df_airports = pd.read_excel(airport_data)
df_fleet = pd.read_excel(fleet_data)
df_demand = pd.read_excel(demand_data, header=None)

# print(df_airports)
# print(df_fleet)
# print(df_demand)

#Defining Parameters Fleet
fleet       = pd.read_excel(fleet_data, skiprows=0, header=0, index_col=0)
speed       = np.array(fleet.iloc[0])
capacity    = np.array(fleet.iloc[1])
TAT         = np.array(fleet.iloc[2])
R           = np.array(fleet.iloc[3])
runway_AC   = np.array(fleet.iloc[4])
Cl          = np.array(fleet.iloc[5])
Cf          = np.array(fleet.iloc[6])
Ch          = np.array(fleet.iloc[7])
cfuel       = np.array(fleet.iloc[8])
fleet       = np.array(fleet.iloc[9])

#Defining Parameters Airport
airport     = pd.read_excel(airport_data, skiprows=0, header=0, index_col=0)
IATA        = np.array(airport.iloc[0])
latitude    = airport.iloc[1].tolist()
longitude   = airport.iloc[2].tolist()
runway_AP   = np.array(airport.iloc[3])

Re = 6371 
f = 1.42
Yield = 0.26

#Defining Sets
n = len(IATA)
N = range(len(IATA))
K = range(len(speed))
T = range(120*10)

def demand(dep, arr, tw):
    return float(df_demand[(df_demand[1] == dep) & (df_demand[2] == arr)][tw+3])

step_demand = np.zeros((2,20,1200))
#initalize hub = 6
for i in N:
    for t in T:
        step_demand[0][i][t] = demand(IATA[3], IATA[i], int(t/40))
for i in N:
    for t in T:
        step_demand[1][i][t] = demand(IATA[i], IATA[3], int(t/40))

#%%
# Haversine Distance Calculation
lat_rad = np.radians(latitude)
lon_rad = np.radians(longitude)

IATA_index = list(IATA)
def distance(dep, arr):
    delta_lat = lat_rad[IATA_index.index(dep)] - lat_rad[IATA_index.index(arr)]
    delta_lon = lon_rad[IATA_index.index(dep)] - lon_rad[IATA_index.index(arr)]
    a = math.sin(delta_lat / 2)**2 + math.cos(lat_rad[IATA_index.index(dep)]) * math.cos(lat_rad[IATA_index.index(arr)]) * math.sin(delta_lon / 2)**2
    sigma = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    distance = Re * sigma
    return distance

#%%
#Define operating_costs
def costs(dep, arr, k):
    if dep == arr:
        operating_costs = 0
    else:
        CT = Ch[k] * (distance(dep, arr) / speed[k])
        CFuel = cfuel[k] * f * distance(dep, arr) / 1.5
        operating_costs = CT + Cf[k] + CFuel  # Zorg ervoor dat Cf[k] wordt gebruikt
    return operating_costs

#Define revenue 
def revenue(dep, arr, k, t):
    flow = 0
    Yield = 0.26  # Zet een standaardwaarde voor Yield
    timew = int(t / 40)
    if dep == IATA[3]:
        if step_demand[0][IATA_index.index(arr)][t] <= capacity[k]:
            flow = step_demand[0][IATA_index.index(arr)][t]
        else:
            flow = capacity[k]
    
    elif arr == IATA[3]:
        if step_demand[1][IATA_index.index(dep)][t] <= capacity[k]:
            flow = step_demand[1][IATA_index.index(dep)][t]
        else:
            flow = capacity[k]
    flow = flow / 1000
    revenue = Yield * distance(dep, arr) * flow
    return revenue

# Define total profit
M = -10000000000 
def total_profit(dep, arr, k, t):
    if R[k] < distance(dep, arr):
        profit = M
    elif runway_AC[k] > runway_AP[IATA_index.index(dep)] or runway_AC[k] > runway_AP[IATA_index.index(arr)]:
        profit = M
    else:
        profit = revenue(dep, arr, k, t) - costs(dep, arr, k)
    return profit 

#%%

#calculate flight times
flight_time = np.zeros((20,3))
for i in N:
    for k in K:
        flight_time[i][k] = np.ceil((distance(IATA[3], IATA[i])/speed[k] + 0.5 + 0.5*TAT[k]/60)*10) 

#%%

#calculate profit per aircraft type

#initialize state arrays
state = np.zeros((20,1200,3))
#initialize control city array
hub_ap = np.zeros((1200,3), dtype=int)
hub_ap[-1,:] = int(3) #begin with the hub

#%%

for k in K:
    #calculate profit for the specific actions
    stay_action = np.zeros((20, 1200))
    to_hub_action = np.zeros((20,1200))

    for t in range(1200):
        for airport in range(0,3):
            to_hub_action[airport][t] = total_profit(IATA[airport], IATA[3], k, t)
        for airport in range(4,20):
            to_hub_action[airport][t] = total_profit(IATA[airport], IATA[3], k, t)
  
    #calculate the profit for flying back to the hub
    from_hub_action = np.zeros((20,1200))
    for t in range(1200):
        for airport in range(0,3):
            from_hub_action[airport][t] = total_profit(IATA[3], IATA[airport], k, t)
        for airport in range(4,20):
            from_hub_action[airport][t] = total_profit(IATA[3], IATA[airport], k, t)
            
    #backwards iteration
    for t in range(1198,-1,-1):
        for a in range(20):
            #if it is the hub
            if a == 3:

                #check all airports
                profits = np.zeros(20)
                for i in range(20):

                    #check if the flight leg fits within time window
                    if t+flight_time[i][k] < 1199:
                        profits[i] = from_hub_action[i][t] + state[i][int(t+flight_time[i][k])][k]

                #choose the airport with the highest profit
                state[a][t][k] = max(profits)
                hub_ap[t][k] = np.argmax(profits).astype(int)

            #if it is not the hub    
            else:
                if t+flight_time[a][k] < 1199:
                    state[a][t][k] = max(to_hub_action[a][t] + state[4][int(t+flight_time[a][k])][k], 
                                      stay_action[a][t+1] + state[a][t+1][k])
                    
#%%
#deduct leasing costs
final_profit = np.zeros(3)
for k in K:
    final_profit[k] = state[3][0][k] - 5 * Cl[k]

print('The aircraft type delivering the highest profit is aircraft type ', np.argmax(final_profit) + 1)
print('with final profit ', max(final_profit))

#%%
#give the flight schedule for the aircraft with the highest profit
# Initialize the flight schedule for each aircraft
flight_schedules = {aircraft: [] for aircraft in range(len(speed))}

# Start flight scheduling
for k in range(len(speed)):  # Iterate over each aircraft type
    current_time = int(0)
    time_steps = 1199
    aircraft_available = True
    at_hub = True
    start_airport = IATA[3]
    total_flight_time = 0

    while current_time < time_steps:
        if at_hub:
            # Start at hub
            start_airport = IATA[3]
            if aircraft_available:
                if hub_ap[int(current_time)][k]== 3:
                    current_time += 1

                # If control city is not the hub
                if hub_ap[int(current_time)][k] != 3:
                    # Check whether round trip fits in time window
                    if current_time + 2 * flight_time[hub_ap[int(current_time)][k]][k] >= time_steps:
                        break

                    # Fly to the control city
                    else:
                        destination = IATA[hub_ap[int(current_time)][k]]
                        destination_index = hub_ap[int(current_time)][k]
                        travel_time = flight_time[hub_ap[int(current_time)][k]][k]

                        # Calculate the flow for this flight
                        demand_served = 0
                        if capacity[k] <= step_demand[0][destination_index][int(current_time)]:
                            demand_served = capacity[k]
                            
                            # Check for unmet demand in the previous time window
                            for offset in [1, 2]:  # Loop over t-1 and t-2
                                previous_timew = int((current_time - 40) / 40)  # Previous time window
                                if previous_timew >= 0:  # Ensure it's within bounds
                                    previous_unsatisfied_demand = step_demand[0][destination_index][previous_timew * 40:(previous_timew + 1) * 40].sum()
                                    max_extra_demand = 0.2 * step_demand[0][destination_index][previous_timew * 40:(previous_timew + 1) * 40].max()
                                    
                                    # Take additional demand from the previous time window if possible
                                    extra_demand = min(max_extra_demand, capacity[k] - demand_served, previous_unsatisfied_demand)
                                    demand_served += extra_demand
                            
                                    # Update previous time window's demand
                                    step_demand[0][destination_index][previous_timew * 40:(previous_timew + 1) * 40] -= extra_demand
                        
                        else:
                            demand_served = step_demand[0][destination_index][int(current_time)]

                        # Update the demand for the current time window
                        window = int(current_time / 40)
                        step_demand[0][destination_index][window * 40:window * 40 + 40] -= demand_served

                        # Store the flight in the schedule
                        flight_schedules[k].append({
                            'start': start_airport,
                            'destination': destination,
                            'departure_time': current_time,
                            'arrival_time': current_time + travel_time,
                            'flow': demand_served,
                        })

                        total_flight_time += travel_time
                        current_time += travel_time
                        at_hub = False
                        aircraft_available = False
            else:
                next_available_time = flight_schedules[k][-1]['arrival_time']
                if current_time >= next_available_time:
                    aircraft_available = True
                else:
                    current_time += 1

        # If not at hub
        else:
            if aircraft_available:
                if state[destination_index][int(current_time + 1)][k] > state[4][int(current_time + 1)][k]:
                    current_time += 1
                else:
                    start_airport = destination
                    travel_time = flight_time[destination_index, k]
                    destination = IATA[3]
                    destination_index = 3

                    # Calculate the flow for this flight
                    demand_served = 0
                    if capacity[k] <= step_demand[0][destination_index][int(current_time)]:
                        demand_served = capacity[k]
                        
                        # Check for unmet demand in the previous time window
                        for offset in [1, 2]:  # Loop over t-1 and t-2
                            previous_timew = int((current_time - 40) / 40)  # Previous time window
                            if previous_timew >= 0:  # Ensure it's within bounds
                                previous_unsatisfied_demand = step_demand[0][destination_index][previous_timew * 40:(previous_timew + 1) * 40].sum()
                                max_extra_demand = 0.2 * step_demand[0][destination_index][previous_timew * 40:(previous_timew + 1) * 40].max()
                                
                                # Take additional demand from the previous time window if possible
                                extra_demand = min(max_extra_demand, capacity[k] - demand_served, previous_unsatisfied_demand)
                                demand_served += extra_demand
                        
                                # Update previous time window's demand
                                step_demand[0][destination_index][previous_timew * 40:(previous_timew + 1) * 40] -= extra_demand
                    
                    else:
                        demand_served = step_demand[0][destination_index][int(current_time)]

                    window = int(current_time / 40)
                    step_demand[1][IATA_index.index(start_airport)][window * 40:window * 40 + 40] -= demand_served

                    # Store the flight in the schedule
                    flight_schedules[k].append({
                        'start': start_airport,
                        'destination': destination,
                        'departure_time': current_time,
                        'arrival_time': current_time + travel_time,
                        'flow': demand_served,
                    })

                    total_flight_time += travel_time
                    current_time += travel_time
                    at_hub = True
                    aircraft_available = False
            else:
                next_available_time = flight_schedules[k][-1]['arrival_time']
                if current_time >= next_available_time:
                    aircraft_available = True
                else:
                    current_time += 1

#%%
#iteration 2
for k in K:
    #calculate profit for the specific actions
    stay_action = np.zeros((20, 1200))
    to_hub_action = np.zeros((20,1200))

    for t in range(1200):
        for airport in range(0,3):
            to_hub_action[airport][t] = total_profit(IATA[airport], IATA[3], k, t)
        for airport in range(4,20):
            to_hub_action[airport][t] = total_profit(IATA[airport], IATA[3], k, t)
  
    #calculate the profit for flying back to the hub
    from_hub_action = np.zeros((20,1200))
    for t in range(1200):
        for airport in range(0,3):
            from_hub_action[airport][t] = total_profit(IATA[3], IATA[airport], k, t)
        for airport in range(4,20):
            from_hub_action[airport][t] = total_profit(IATA[3], IATA[airport], k, t)
            
    #backwards iteration
    for t in range(1198,-1,-1):
        for a in range(20):
            #if it is the hub
            if a == 3:

                #check all airports
                profits = np.zeros(20)
                for i in range(20):

                    #check if the flight leg fits within time window
                    if t+flight_time[i][k] < 1199:
                        profits[i] = from_hub_action[i][t] + state[i][int(t+flight_time[i][k])][k]

                #choose the airport with the highest profit
                state[a][t][k] = max(profits)
                hub_ap[t][k] = np.argmax(profits).astype(int)

            #if it is not the hub    
            else:
                if t+flight_time[a][k] < 1199:
                    state[a][t][k] = max(to_hub_action[a][t] + state[3][int(t+flight_time[a][k])][k], 
                                      stay_action[a][t+1] + state[a][t+1][k])
            

#deduct leasing costs
final_profit = np.zeros(3)
for k in K:
    final_profit[k] = state[3][0][k] - 5 * Cl[k]

print('The aircraft type delivering the highest profit is aircraft type ', np.argmax(final_profit) + 1)
print('with final profit ', max(final_profit))

#%%

print(total_flight_time)

#%%
def adjusted_arrival_time(arrival_time, k):
    # Adjust the arrival time by subtracting TAT and 30 minutes
    return arrival_time - (TAT[k] / 60) - 0.5

# Print the flight schedule for each aircraft type
for aircraft, schedule in flight_schedules.items():
    print(f"Flight schedule for aircraft type {aircraft + 1}:")
    for flight in schedule:
        adjusted_arrival = adjusted_arrival_time(flight['arrival_time'], aircraft)
        print(f"  Start: {flight['start']}, Destination: {flight['destination']}, "
              f"Departure: {flight['departure_time']}," f"Arrival: {(adjusted_arrival)},"
              f"Flow: {flight['flow']}")

