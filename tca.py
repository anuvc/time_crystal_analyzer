import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import cwt, ricker, hilbert
from scipy.signal import find_peaks
from scipy import fft
from scipy.signal import butter, filtfilt

class TimeCrystalAnalyzer:
    def __init__(self, sampling_rate=1000):
        self.sampling_rate = sampling_rate
        self.dt = 1/sampling_rate
        
    def butter_bandpass(self, lowcut, highcut, order=5):
        """Create Butterworth bandpass filter"""
        nyq = 0.5 * self.sampling_rate
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        return b, a

    def find_dominant_frequencies(self, signal, time):
        """
        Improved frequency detection using FFT
        """
        # Compute FFT
        n = len(signal)
        yf = fft.fft(signal)
        xf = fft.fftfreq(n, self.dt)
        
        # Look at positive frequencies only
        positive_freqs = xf[:n//2]
        magnitude_spectrum = 2.0/n * np.abs(yf[:n//2])
        
        # Find peaks with better parameters
        peaks, properties = find_peaks(magnitude_spectrum,
                                     height=0.05*np.max(magnitude_spectrum),
                                     distance=int(n/100),  # Minimum distance between peaks
                                     prominence=0.05*np.max(magnitude_spectrum))
        
        # Sort peaks by prominence
        sorted_idx = np.argsort(properties['prominences'])[::-1]
        peaks = peaks[sorted_idx]
        
        # Return frequencies and their magnitudes
        dominant_freqs = positive_freqs[peaks]
        magnitudes = magnitude_spectrum[peaks]
        
        return dominant_freqs[:5], magnitudes[:5]  # Return top 5 frequencies

    def extract_components(self, signal, frequencies):
        """
        Extract frequency components using improved bandpass filtering
        """
        components = {}
        
        for freq in frequencies:
            # Wider bandwidth for higher frequencies
            bandwidth = max(0.5, freq * 0.2)  # 20% of frequency or 0.5 Hz, whichever is larger
            lowcut = freq - bandwidth/2
            highcut = freq + bandwidth/2
            
            # Apply bandpass filter
            b, a = self.butter_bandpass(lowcut, highcut)
            filtered = filtfilt(b, a, signal)
            
            components[freq] = filtered
            
        return components
    
    def analyze_phases(self, components):
        """
        Improved phase relationship analysis
        """
        phases = {}
        reference_freq = min(components.keys())
        ref_analytic = hilbert(components[reference_freq])
        ref_phase = np.unwrap(np.angle(ref_analytic))
        
        for freq, component in components.items():
            if freq != reference_freq:
                analytic = hilbert(component)
                phase = np.unwrap(np.angle(analytic))
                relative_phase = phase - ref_phase
                phases[freq] = relative_phase % (2*np.pi)
        
        return phases
    
    def analyze_signal(self, signal, time):
        """
        Main analysis function
        """
        # Find dominant frequencies
        frequencies, magnitudes = self.find_dominant_frequencies(signal, time)
        print("Detected frequencies (Hz):", frequencies)
        print("Relative magnitudes:", magnitudes / np.max(magnitudes))
        
        # Extract components
        components = self.extract_components(signal, frequencies)
        
        # Analyze phases
        phases = self.analyze_phases(components)
        
        return {
            'frequencies': frequencies,
            'magnitudes': magnitudes,
            'components': components,
            'phases': phases
        }
    
    def plot_analysis(self, signal, time, results):
        """
        Enhanced visualization
        """
        plt.figure(figsize=(15, 12))
        
        # Plot original signal
        plt.subplot(411)
        plt.plot(time, signal)
        plt.title('Original Signal')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.grid(True)
        
        # Plot components
        plt.subplot(412)
        for freq, component in results['components'].items():
            plt.plot(time, component, label=f'{freq:.1f} Hz')
        plt.title('Extracted Components')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.legend()
        plt.grid(True)
        
        # Plot phase relationships
        plt.subplot(413)
        for freq, phase in results['phases'].items():
            plt.plot(time, phase, label=f'Phase rel. to {min(results["frequencies"]):.1f} Hz: {freq:.1f} Hz')
        plt.title('Phase Relationships')
        plt.xlabel('Time (s)')
        plt.ylabel('Phase (radians)')
        plt.legend()
        plt.grid(True)
        
        # Polar plot of phases at middle time point
        ax = plt.subplot(414, projection='polar')
        mid_point = len(time) // 2
        for freq, phase in results['phases'].items():
            magnitude = results['magnitudes'][list(results['frequencies']).index(freq)]
            ax.scatter(phase[mid_point], magnitude, label=f'{freq:.1f} Hz', s=100)
        ax.set_title('Phase Relationships (Polar)')
        plt.legend()
        
        plt.tight_layout()
        plt.show()
        
        # Additional frequency analysis plot
        plt.figure(figsize=(10, 5))
        n = len(signal)
        yf = fft.fft(signal)
        xf = fft.fftfreq(n, self.dt)
        plt.plot(xf[:n//2], 2.0/n * np.abs(yf[:n//2]))
        plt.grid(True)
        plt.title('Frequency Spectrum')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Magnitude')
        plt.show()

# Example usage
if __name__ == "__main__":
    # Generate test signal with multiple components
    t = np.linspace(0, 10, 10000)
    signal = (np.sin(2*np.pi*5*t) + 
             0.5*np.sin(2*np.pi*20*t) + 
             0.3*np.sin(2*np.pi*35*t) + 
             np.random.normal(0, 0.1, len(t)))
    
    analyzer = TimeCrystalAnalyzer(sampling_rate=1000)
    results = analyzer.analyze_signal(signal, t)
    analyzer.plot_analysis(signal, t, results)