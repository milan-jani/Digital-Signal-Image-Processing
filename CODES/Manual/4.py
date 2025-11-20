#AIM: Write a python program to Simulate continuous and discrete ramp signal,
import numpy as np
import matplotlib.pyplot as plt

def simulate_continuous_ramp(time, slope):
    ramp = np.zeros_like(time)
    ramp[time >= 0] = slope * time[time >= 0]
    return ramp

def simulate_discrete_ramp(num_samples, slope):
    ramp = np.zeros(num_samples)
    ramp[num_samples // 2:] = slope * np.arange(num_samples // 2, num_samples)
    return ramp

# Define the time range for the continuous ramp signal
time = np.linspace(-5, 5, 1000)  # Time range from -5 to 5

# Define the number of samples and slope for the discrete ramp signal
num_samples = 20  # Number of samples
slope = 2  # Slope of the ramp

# Simulate the continuous ramp signal
continuous_ramp = simulate_continuous_ramp(time, slope)

# Simulate the discrete ramp signal
discrete_ramp = simulate_discrete_ramp(num_samples, slope)

# Plot and display the continuous and discrete ramp signals
plt.figure(figsize=(10, 6))
plt.subplot(2, 1, 1)
plt.plot(time, continuous_ramp)
plt.title('Continuous Ramp Signal')
plt.xlabel('Time')
plt.ylabel('Amplitude')

plt.subplot(2, 1, 2)
plt.stem(discrete_ramp)
plt.title('Discrete Ramp Signal')
plt.xlabel('Sample')
plt.ylabel('Amplitude')

plt.tight_layout()
plt.show()

# Save the ramp signal arrays (optional)
# np.savetxt('continuous_ramp.txt', continuous_ramp, delimiter=',')
# np.savetxt('discrete_ramp.txt', discrete_ramp, delimiter=',')
# Save the plots as image files
plt.savefig('continuous_ramp_signal.png', dpi=300, bbox_inches='tight') 
plt.savefig('discrete_ramp_signal.png', dpi=300, bbox_inches='tight')
# plt.savefig('ramp_signal.png', dpi=300, bbox_inches='tight')