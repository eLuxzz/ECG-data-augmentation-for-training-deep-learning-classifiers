import numpy as np 

class DataAugmenter():
    def __init__(self):
        pass
    
    def add_gaussian_noise(self, signalData, mean=0, std=0.05):
        signalData = np.squeeze(signalData)
        noise = np.random.normal(mean, std, signalData.shape)
        augmented_data = signalData + noise
        return augmented_data
    
    def add_baseline_wander(self, signalData, frequency=0.5, amplitude=0.1, sampling_rate=500):
        signalData = np.squeeze(signalData)
        t = np.arange(len(signalData)) / sampling_rate
        baseline_noise = amplitude * np.sin(2 * np.pi * frequency * t)
        baseline_noise = baseline_noise[:, None]  # Alternativ metod
        augmented_data = signalData + baseline_noise
        return augmented_data
    
    def add_powerline_noise(self, signalData, frequency=50, amplitude=0.01, sampling_rate=500):
        signalData = np.squeeze(signalData)
        t = np.arange(len(signalData)) / sampling_rate
        powerline_noise = amplitude * np.sin(2 * np.pi * frequency * t)
        powerline_noise = powerline_noise[:, None]
        augmented_data = signalData + powerline_noise
        return augmented_data