import threading
import random
import time
from collections import defaultdict

# A lock to prevent multiple threads from writing to the stats dictionary at the same time
stats_lock = threading.Lock()

# This dictionary will hold the stats for each flow
stats = defaultdict(lambda: {'packets': 0, 'bytes': 0, 'label': 0})

# Total simulation time
TOTAL_SIMULATION_TIME = 60  # seconds

# Define benign and malicious flow rates (packets per second)
BENIGN_FLOW_RATE = 5
MALICIOUS_FLOW_RATE = 100  # Much higher rate to simulate a DDoS attack

def simulate_flow(flow_id, packet_rate, duration):
    start_time = time.time()
    while time.time() - start_time < duration:
        # Simulate sending packets based on the packet_rate
        time.sleep(1 / packet_rate)
        packet_size = random.randint(40, 1500)  # Random packet size
        with stats_lock:
            stats[flow_id]['packets'] += 1
            stats[flow_id]['bytes'] += packet_size
            # Determine if the flow is benign (0) or malicious (1) based on the packet rate
            stats[flow_id]['label'] = 1 if packet_rate >= MALICIOUS_FLOW_RATE else 0

# Start benign flows
for i in range(10):
    flow_id = (f'192.168.0.{i}', 80)  # Benign flows to port 80
    threading.Thread(target=simulate_flow, args=(flow_id, BENIGN_FLOW_RATE, TOTAL_SIMULATION_TIME)).start()

# Start malicious flows
for i in range(10, 20):
    flow_id = (f'192.168.0.{i}', 80)  # Malicious flows to port 80
    threading.Thread(target=simulate_flow, args=(flow_id, MALICIOUS_FLOW_RATE, TOTAL_SIMULATION_TIME)).start()

# Wait for the simulation to run
time.sleep(TOTAL_SIMULATION_TIME)

# Print the collected stats
with stats_lock:
    for flow, data in stats.items():
        print(f"Flow {flow} - Packets: {data['packets']}, Bytes: {data['bytes']}, Label: {data['label']}")
