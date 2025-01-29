# Initial position, velocity, and acceleration
position = 0
velocity = 5
acceleration = 2

# Time step for simulation
time_step = 2

# Number of time steps to simulate
num_steps = 10

# Main simulation loop
for i in range(num_steps):
    # Update position based on velocity and acceleration
    position = position + (velocity * time_step) + (0.5 * acceleration * (time_step ** 2))
    
    # Update velocity based on acceleration
    velocity = velocity + (acceleration * time_step)
    
    # Print current position and velocity
    print(f"Time: {i}, Position: {position}, Velocity: {velocity}")
