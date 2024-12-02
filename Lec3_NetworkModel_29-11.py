from gurobipy import *

Airports = ['A1', 'A2', 'A3']
N = range(len(Airports))
Aircrafts = ['AC1', 'AC2', 'AC3','AC4']
K = range(len(Aircrafts))
Cx = [300, 600, 1250, 2000]
ct = [750, 775, 1400, 2800]
cf = [1, 2, 3.75, 9]
Cl = [15000, 34000, 80000, 190000]
LF = 0.75
s = [45, 70, 150, 320]
sp = [550, 820, 850, 870]
LTO = 20/60
BT = 10 * 7
V = [550, 820, 850, 870]
# AC = 2
y = 0.18  # yield
q = [[0, 1000, 200],
      [1000, 0, 300],
      [200, 300, 0]]

d = [[0, 2236, 3201],
     [2236, 0, 3500],
     [3201, 3500, 0]]
f = 1.42
hub = Airports[0] 
R = [1500, 3300, 6300, 12000]
RW_AC = [1400, 1600, 1800, 2600]
# RW_AP = airport_data[7] # connect airport data runway
TAT_nohub = [25, 35, 45, 60]
TAT_hub = [tat * 1.5 for tat in TAT_nohub]



Ct={}
for i in N:
    for j in N:
        if i != j:
            for k in K:
                Ct[i, j, k] = ct[k] * d[i][j] / V[k] 

Cf={}
for i in N:
    for j in N:
        if i != j:
            for k in K:
                Cf[i,j,k] = ((cf[k] * f) / 1.5) * d[i][j]
                
# C={}
# for i in N:
#     for j in N:
#         if i!= j:
#             for k in K:
#                 if i == hub or j == hub:
#                     C[i,j,k] = (Cx[k] + Ct[i,j,k] + Cf[i,j,k]) * 0.7
#                 else:
#                     C[i,j,k] = Cx[k] + Ct[i,j,k] + Cf[i,j,k]

C = {}
for i in N:
    for j in N:
        if i != j:
            for k in K:
                if Airports[i] == hub or Airports[j] == hub:
                    C[i,j,k] = (Cx[k] + Ct[i,j,k] + Cf[i,j,k]) * 0.7 
                else:
                    C[i,j,k] = Cx[k] + Ct[i,j,k] + Cf[i,j,k] 
                    
# Lease costs need to be taken into account

g = {}  # Binary variabelen voor hubs
for i, airport in enumerate(Airports):
    g[i] = 0 if airport == hub else 1  # 0 voor hub, 1 voor andere luchthavens

# # Calculate the Available Seat Kilometers (ASK)
# ASK = {}
# for k in K:
#     # Assuming each aircraft flies between all airports for simplicity
#     total_distance = sum([d[i][j] for i in N for j in N if i != j])  # Total distance flown by the aircraft
#     total_seat_km = s[k] * total_distance * LF  # Seat kilometers
#     ASK[k] = total_seat_km

# # Calculate CASK for each aircraft
# CASK = {}
# for k in K:
#     CASK[k] = C[k] / ASK[k]  # CASK for the aircraft

# # Print CASK for each aircraft
# for k in K:
#     print(f"CASK for {Aircrafts[k]}: {CASK[k]}")
    
# #CASK = C[i,j,k]/(d[i][j]*s[k])
# print (CASK)

# Start modelling optimization problem
m = Model('practice')

# Decision Variables
x = {}
for i in N:
    for j in N:
        x[i,j] = m.addVar(lb=0, vtype=GRB.INTEGER)

z = {}
for i in N:
    for j in N:
        for k in K:
            z[i,j,k] = m.addVar(lb=0, vtype=GRB.INTEGER)
        
w = {}
for i in N:
    for j in N:
        w[i,j] = m.addVar(lb=0, vtype=GRB.INTEGER)

AC = {}
for k in K:
    AC[k] = m.addVar(lb=0, vtype=GRB.INTEGER)
    
m.update()

# Objective Function: Maximize Revenue - Cost
# m.setObjective(quicksum(y * distance[i][j] * x[i, j] - (quicksum(CASK[k] * distance[i][j] * s[k] * z[i, j, k]) for k in K)
#                         for i in N for j in N), GRB.MAXIMIZE)

m.setObjective(
    quicksum(y * d[i][j] * x[i, j] for i in N for j in N if i != j) 
    - quicksum(CASK[k] * distance[i][j] * s[k] * z[i, j, k] for i in N for j in N for k in K if i != j)
)

# Constraints
# 1. Flow should not exceed demand
for i in n:
    for j in n:
        m.addConstr(x[i, j] + w[i,j] <= q[i][j], name=f"DemandConstraint_{i}_{j}")  # C1

# 1*. Only consider transfer passangers if the hub is not origin or destination        
for i in n:
    for j in n:
        m.addConstr(w[i,j] <= q[i][j] * g[i] * g[j], name=f"TransferPassangers_{i}_{j}") #C1*

# 2. Flow is limited by capacity
for i in n:
    for j in n:
        m.addConstr(x[i, j] <= (quicksum(z[i,j,k] * s[k] * LF) for k in K), name=f"CapacityConstraint_{i}_{j}")  # C2
        
# Constraint: x_ij + sum(w_im * (1 - g_j)) + sum(w_mj * (1 - g_i)) <= z_ij * s * LF
for i in n:
    for j in n:
        m.addConstr(x[i,j] + quicksum(w[(i,m)] * (1 - g[j]) for m in n) + quicksum(w[(m,j)] * (1 - g[i]) for m in n) <= z[i,j] * s * LF, name=f"constraint_{Airports[i]}_{Airports[j]}")

# 3. Flow conservation: flow in equals flow out for each airport
for i in n:
    for k in K:
        m.addConstr(quicksum(z[i, j, k] for j in n) == quicksum(z[j, i, k] for j in n), 
                name=f"FlowConservation_{i}")  # C3

# 4. Total time constraint for the flights
for k in K:
    m.addConstr(quicksum(quicksum((distance[i][j] / sp[k]+ LTO) * z[i, j, k] for i in n) for j in n) <= BT[k] * AC[k], 
            name="TimeConstraint")  # C4
    
# 5. Aircraft range used to define matrix akij and constrain frequency to range limits
# Define ak_ij
ak = {}
for i in N:
    for j in N:
        if i != j:
            for k in K:
                if d[i][j] <= R[k]:  # Check if the aircraft range is sufficient
                    ak[i, j, k] = 10000  # Valid range
                else:
                    ak[i, j, k] = 0  # Invalid range

# Add constraint z_ij <= ak_ij
for i in N:
    for j in N:
        if i != j:
            for k in K:
                m.addConstr(z[i, j, k] <= ak[i, j, k], name=f"range_constraint_{i}_{j}_{k}") # C5

# 6. Runway airport must be bigger or equal to runway aircraft

# 7. Turn-Around-Time

# Solve the model
m.optimize()

# Check the status of the optimization
status = m.status

if status == GRB.Status.UNBOUNDED:
    print('The model cannot be solved because it is unbounded')
elif status == GRB.Status.OPTIMAL:
    f_objective = m.objVal
    print('***** RESULTS ******')
    print('\nObjective Function Value: \t %g' % f_objective)
elif status != GRB.Status.INF_OR_UNBD and status != GRB.Status.INFEASIBLE:
    print('Optimization was stopped with status %d' % status)

# Print out Solutions for frequencies (z[i,j])
print("\nFrequencies:----------------------------------")
for i in n:
    for j in n:
        if z[i, j].X > 0:
            print(f"{Airports[i]} to {Airports[j]}: {z[i, j].X}")
