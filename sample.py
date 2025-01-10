import matplotlib.pyplot as plt
import numpy as np

# Simulated accuracy values for illustration purposes
scenarios = ['Clean Images', 'Adversarial Attacks (FGSM)', 'Defended (Adversarial Training)',
             'Defended (Gaussian Noise)', 'Defended (Denoising Autoencoder)']

# Simulated accuracies (percentages)
accuracy = [92, 45, 85, 78, 82]

# Model name
model_name = 'ResNet-50'

# Plotting the simulated results
plt.figure(figsize=(10, 6))
plt.bar(scenarios, accuracy, color=['#4CAF50', '#F44336', '#2196F3', '#FFC107', '#9C27B0'])
plt.title(f'Adversarial Attacks and Defenses in Medical Imaging for {model_name}')
plt.ylabel('Accuracy (%)')
plt.xlabel('Scenarios')
plt.ylim(0, 100)

# Annotating accuracy values on top of bars
for i, val in enumerate(accuracy):
    plt.text(i, val + 2, f"{val}%", ha='center', va='bottom', fontweight='bold')

# Display the plot
plt.xticks(rotation=15)
plt.tight_layout()
plt.show()