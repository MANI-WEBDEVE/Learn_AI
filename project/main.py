import numpy as np
import matplotlib.pyplot as plt

# Parameters
frequency = 28 # Frequency in Hz
sampling_rate = 43100   # Sampling rate in Hz
duration = 2  # Duration in seconds

# Generate time points
t = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False)

# Generate sine wave
y = np.sin(2 * np.pi * frequency * t)

# Plot the sine wave
plt.plot(t, y)
plt.title(f'Sine Wave with Frequency {frequency} Hz')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')
plt.grid(True)
plt.show()

import pyaudio


# Parameters
frequency = 28 # Frequency in Hz (A4 note)
duration = 20 # Duration in seconds
sampling_rate = 43100  # Sampling rate in Hz

# Generate samples
t = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False)
samples = (np.sin(2 * np.pi * frequency * t)).astype(np.float32)

# Initialize PyAudio
p = pyaudio.PyAudio()

# Open a stream
stream = p.open(format=pyaudio.paFloat32, channels=1, rate=sampling_rate, output=True)

# Play the sound
stream.write(samples.tobytes())

# Close the stream
stream.stop_stream()
stream.close()
p.terminate()
