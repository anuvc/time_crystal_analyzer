import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import cwt, ricker

# Step 1: Load or generate a complex signal
# Generate a sample signal for testing purposes
time = np.linspace(0, 1, 1000, endpoint=False)
signal = np.sin(2 * np.pi * 5 * time) + np.sin(2 * np.pi * 20 * time) + np.random.normal(0, 0.2, len(time))

# Plot the signal
plt.figure(figsize=(10, 4))
plt.plot(time, signal)
plt.title("Original Complex Signal")
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.show()

# Step 2: Decompose the signal using wavelet transform
# Define scales for the wavelet transform
scales = np.arange(1, 128)

# Use Ricker (Mexican hat) wavelet
coefficients = cwt(signal, ricker, scales)

# Visualize the wavelet transform coefficients
plt.figure(figsize=(10, 6))
plt.imshow(np.abs(coefficients), extent=[0, 1, scales[-1], scales[0]], cmap='viridis', aspect='auto')
plt.colorbar(label='Magnitude')
plt.title("Wavelet Transform Coefficients")
plt.xlabel("Time")
plt.ylabel("Scale")
plt.show()

# Select specific scales to visualize the wavelets
selected_scales = [5, 20, 50]

# Extract the wavelets at selected scales
for scale in selected_scales:
    wavelet = coefficients[scale - 1]  # Scales are 1-indexed; array is 0-indexed
    plt.figure(figsize=(10, 4))
    plt.plot(time, wavelet)
    plt.title(f"Wavelet at Scale {scale}")
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.show()

# Visualize the signal as a time crystal

# Step 1: Define the host circle
host_radius = 1.0  # Base radius of the largest host circle
theta = np.linspace(0, 2 * np.pi, len(signal))  # Angles for the host circle

# Step 2: Extract wavelet information for nested circles
selected_scales = [5, 20, 50]  # Scales to visualize (larger scale = larger circle)
nested_circles = []

for scale in selected_scales:
    wavelet = coefficients[scale - 1]  # Wavelet coefficients for the scale
    amplitude = np.abs(wavelet)  # Use magnitude for circle size
    phase = np.angle(wavelet)  # Phase for positioning

    # Normalize amplitude for visualization
    normalized_amplitude = amplitude / np.max(amplitude)

    # Define circle properties for each phase
    for i in range(len(phase)):
        intersection_angle = phase[i]  # Phase determines intersection point
        radius = host_radius * (1 / scale)  # Scale down radius for smaller scales
        center_x = (host_radius - radius) * np.cos(intersection_angle)
        center_y = (host_radius - radius) * np.sin(intersection_angle)

        # Append data for nested circle
        nested_circles.append((center_x, center_y, radius))

# Step 3: Plot the host and nested circles
fig, ax = plt.subplots(figsize=(8, 8))

# Plot the largest host circle
ax.add_artist(plt.Circle((0, 0), host_radius, color="black", fill=False, linewidth=1))

# Plot the nested guest circles
for center_x, center_y, radius in nested_circles:
    circle = plt.Circle((center_x, center_y), radius, color="black", fill=False, linewidth=1)
    ax.add_artist(circle)

# Step 4: Beautify the plot
ax.set_xlim(-1.5, 1.5)
ax.set_ylim(-1.5, 1.5)
ax.set_aspect('equal', adjustable='datalim')
ax.set_title("Nested Time Crystal Visualization")
plt.grid(True)
plt.show()