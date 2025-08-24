LSM from Scratch: Demonstrating Noise Resilience
This project provides a simple, from-scratch implementation of a Liquid State Machine (LSM) to demonstrate one of its key advantages: improving classification accuracy on noisy temporal data.

By running two parallel classification tasks—one directly on noisy data and one on the same data after being processed by an LSM—this script clearly shows the LSM's ability to act as a powerful pre-processor and noise filter.

🧠 What is a Liquid State Machine (LSM)?
An LSM is a type of reservoir computer. Think of it as a "liquid" or randomly connected recurrent neural network (the reservoir) that gets "perturbed" by an input signal.

Input Signal: A time-series signal is fed into the reservoir.

Reservoir Dynamics: The signal causes complex, dynamic patterns of activation within the reservoir's neurons. This rich, high-dimensional representation of the input's history is called the "liquid state."

Readout: A simple, trainable classifier (like a logistic regression model) is trained not on the original noisy signal, but on the much richer and more separable "liquid states" from the reservoir.

The key idea is that the reservoir itself is fixed (not trained). Only the simple readout classifier is trained. This makes LSMs efficient and particularly good at processing complex temporal patterns, even in the presence of noise.

🎯 Project Goal
The goal is to prove that running a noisy signal through an LSM makes it easier to classify. We do this by comparing the accuracy of two approaches:

Direct Classification:
Noisy Signal -> Classifier -> Low Accuracy

LSM-Enhanced Classification:
Noisy Signal -> LSM Reservoir -> Liquid States -> Classifier -> High Accuracy

🔧 How It Works
Data Generation: The script generates two simple, distinct classes of sine waves.

Noise Injection: A significant amount of noise is added to these sine waves, making them difficult to distinguish visually and for a standard classifier.

LSM Processing: The noisy signals are passed through the LSM, which generates a sequence of "liquid states" (the activation patterns of the reservoir neurons over time).

Classification & Comparison:

A classifier is trained and tested directly on the noisy sine waves.

Another classifier is trained and tested on the LSM's liquid states.

Evaluation: The script prints the classification accuracy for both methods. You will observe that the accuracy of the classifier using the LSM's output is dramatically higher.

🚀 How to Run
To run the demonstration, simply execute the Python script from your terminal:

python lsm_from_scratch.py

📊 Expected Results
When you run the script, you will see output similar to this, clearly showing the performance boost from the LSM:

Classification accuracy without LSM: 54.5%
Classification accuracy with LSM: 98.2%

(Note: Exact percentages may vary slightly between runs due to random initializations.)

This result demonstrates that the LSM successfully filtered the noise and transformed the input data into a format that was much easier for the classifier to understand, leading to a massive improvement in accuracy.
