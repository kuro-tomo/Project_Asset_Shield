import time
import random
import os

# DATA GENERATOR (PRODUCER)
def generate_mock_data():
    price = 3500.0
    log_filename = "mock.log"
    print(f"--- GENERATOR ACTIVE: Writing to {log_filename} ---")
    
    while True:
        price += random.uniform(-10.0, 10.0)
        timestamp = time.time()
        
        # Write to file and force sync to disk
        with open(log_filename, "a") as f:
            f.write(f"{timestamp},{price}\n")
            f.flush()
            os.fsync(f.fileno())
            
        print(f"PUBLISHED: {price:.2f}")
        time.sleep(0.5)

if __name__ == "__main__":
    generate_mock_data()