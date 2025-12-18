import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, bilinear, freqz, cheby1

def design_butterworth_filter(filter_order, cutoff_frequency, sampling_frequency):
    # Design the analog Butterworth filter
    analog_b, analog_a = butter(filter_order, cutoff_frequency, analog=True, btype='low')

    # Perform the bilinear transformation
    digital_b, digital_a = bilinear(analog_b, analog_a, sampling_frequency)

    return digital_b, digital_a

def design_chebyshev_filter(filter_order, cutoff_frequency, sampling_frequency, ripple):
    # Design the analog Chebyshev filter
    analog_b, analog_a = cheby1(filter_order, ripple, cutoff_frequency, analog=True, btype='low')

    # Perform the bilinear transformation
    digital_b, digital_a = bilinear(analog_b, analog_a, sampling_frequency)

    return digital_b, digital_a

def plot_filter_response(butter_b, butter_a, cheby_b, cheby_a, sampling_frequency):
    # Create a single figure with 2x2 subplots
    plt.figure(figsize=(15, 10))
    
    # Butterworth Filter - Magnitude Response
    frequency_b, magnitude_response_b = freqz(butter_b, butter_a, fs=sampling_frequency)
    plt.subplot(2, 2, 1)
    plt.plot(frequency_b, np.abs(magnitude_response_b))
    plt.title('Butterworth Filter Magnitude Response')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.grid(True)
    
    # Butterworth Filter - Impulse Response
    _, impulse_response_b = freqz(butter_b, butter_a, fs=sampling_frequency, worN=512)
    plt.subplot(2, 2, 2)
    plt.plot(np.real(impulse_response_b))
    plt.title('Butterworth Filter Impulse Response')
    plt.xlabel('Samples')
    plt.ylabel('Amplitude')
    plt.grid(True)
    
    # Chebyshev Filter - Magnitude Response
    frequency_c, magnitude_response_c = freqz(cheby_b, cheby_a, fs=sampling_frequency)
    plt.subplot(2, 2, 3)
    plt.plot(frequency_c, np.abs(magnitude_response_c))
    plt.title('Chebyshev Filter Magnitude Response')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.grid(True)
    
    # Chebyshev Filter - Impulse Response
    _, impulse_response_c = freqz(cheby_b, cheby_a, fs=sampling_frequency, worN=512)
    plt.subplot(2, 2, 4)
    plt.plot(np.real(impulse_response_c))
    plt.title('Chebyshev Filter Impulse Response')
    plt.xlabel('Samples')
    plt.ylabel('Amplitude')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

# Specify the desired filter specifications
filter_order = 4  # Filter order
cutoff_frequency = 1000  # Cutoff frequency in Hz
sampling_frequency = 8000  # Sampling frequency in Hz
ripple = 0.5  # Ripple factor for Chebyshev filter

# Design the Butterworth filter
butter_b, butter_a = design_butterworth_filter(filter_order, cutoff_frequency, sampling_frequency)

# Design the Chebyshev filter
cheby_b, cheby_a = design_chebyshev_filter(filter_order, cutoff_frequency, sampling_frequency, ripple)

# Plot both filters' responses in a single 2x2 subplot
plot_filter_response(butter_b, butter_a, cheby_b, cheby_a, sampling_frequency)

# Save the filter coefficients (optional)
filter_path = 'filter_coefficients.txt'
np.savetxt(filter_path, np.vstack((cheby_b, cheby_a)), delimiter=',')
print(f"Filter coefficients saved at: {filter_path}")

# Conclusion
# Butterworth filter: Smooth rolloff, no ripples, better phase response
# Chebyshev filter: Sharp cutoff, faster rolloff, but has passband ripples
# Both filters are stable with decaying impulse responses
# Choice depends on requirement: flat response (Butterworth) vs sharp cutoff (Chebyshev)
