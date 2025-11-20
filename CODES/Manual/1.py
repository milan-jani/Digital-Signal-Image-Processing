#To plot unit impulse signal using Python.
import numpy as np
import matplotlib.pyplot as plt
# Define the time range
t = np.arange(-5, 6, 1)  # from -5 to 5 with step 1     
# Define the unit impulse signal
x = np.zeros_like(t)
x[t == 0] = 1  # Set the value at t=0 to 1
# Plot the unit impulse signal
plt.stem(t, x)
plt.title('Unit Impulse Signal')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.grid()
plt.xlim(-5, 5)
plt.ylim(-0.5, 1.5)
plt.axhline(0, color='black', lw=0.5)  # Add x-axis

plt.axvline(0, color='black', lw=0.5)  # Add y-axis
plt.show()
# Save the plot as an image file
plt.savefig('unit_impulse_signal.png', dpi=300, bbox_inches='tight')