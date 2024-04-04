import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy.linalg import inv

class KalmanFilter:
    def __init__(self, F, H, Q, R):
        self.F = F  # State transition matrix
        self.H = H  # Measurement matrix
        self.Q = Q  # Process noise covariance
        self.R = R  # Measurement noise covariance
        self.P = np.eye(F.shape[0])  # Initial state covariance
        self.x = np.zeros((F.shape[0], 1))  # Initial state

    def predict(self):
        # Predict state and covariance
        self.x = np.dot(self.F, self.x)
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q

    def update(self, z):
        # Update step
        y = z - np.dot(self.H, self.x)
        S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R

        # Kalman Gain
        K = np.dot(np.dot(self.P, self.H.T), inv(S))

        # Update state and covariance
        self.x = self.x + np.dot(K, y)
        self.P = self.P - np.dot(np.dot(K, self.H), self.P)

    def compute_association_probabilities(self, S):
        # Compute association probabilities using Mahalanobis distance
        num_measurements = self.H.shape[0]
        det_S = np.linalg.det(S)
        inv_S = np.linalg.inv(S)
        association_probabilities = np.zeros(num_measurements)
        for i in range(num_measurements):
            d = np.dot(np.dot((self.H[i] - np.dot(self.H[i], self.x)).T, inv_S), (self.H[i] - np.dot(self.H[i], self.x)))
            association_probabilities[i] = np.exp(-0.5 * d) / ((2 * np.pi) ** (self.H.shape[0] / 2) * np.sqrt(det_S))
        return association_probabilities

def main():
    # Read data from CSV file, read only the specified columns
    data = pd.read_csv("test.csv", usecols=[10, 11, 12, 13])

    # Extract data into separate arrays
    measurements = data.values

    # Define state transition matrix
    F = np.array([
        [1, 0, 0, 1],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])

    # Define measurement matrix
    H = np.eye(4)  # Identity matrix since measurement directly reflects state

    # Define process noise covariance matrix
    Q = np.eye(4) * 0.1  # Process noise covariance

    # Define measurement noise covariance matrix
    R = np.eye(4) * 0.01  # Measurement noise covariance, adjusted variance

    # Print input matrices
    print("State Transition Matrix (F):")
    print(F)
    print("\nMeasurement Matrix (H):")
    print(H)
    print("\nProcess Noise Covariance Matrix (Q):")
    print(Q)
    print("\nMeasurement Noise Covariance Matrix (R):")
    print(R)

    # Initialize Kalman filter
    kf = KalmanFilter(F, H, Q, R)

    # Lists to store predicted values for all variables
    predicted_states = []

    # Predict and update for each measurement
    for i, z in enumerate(measurements, start=1):
        # Predict
        kf.predict()

        # Update with measurement
        kf.update(z[:, np.newaxis])

        # Get predicted state
        predicted_state = kf.x.squeeze()

        # Append predicted state
        predicted_states.append(predicted_state)

        # Print predicted state and covariance
        print(f"Measurement {i}:")
        print("Predicted State:")
        print(predicted_state)
        print("\nPredicted Covariance:")
        print(kf.P)
        print()  # Add an empty line for separation

    # Convert predicted_states to numpy array
    predicted_states = np.array(predicted_states)

    # Plotting
    plt.figure(figsize=(8, 6))

    # Plot measured and predicted values against time
    time_steps = np.arange(1, len(predicted_states) + 1)
    labels = ['Range', 'Azimuth', 'Elevation', 'Time']
    for i in range(4):
        plt.plot(time_steps, measurements[:, i], label=f'Measured {labels[i]}', marker='o')
        plt.plot(time_steps, predicted_states[:, i], label=f'Predicted {labels[i]}', linestyle='--', marker='o')
    plt.xlabel('Time (measurement index)')
    plt.ylabel('Value')
    plt.title('Measured and Predicted Values vs. Time')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()
