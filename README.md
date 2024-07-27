This repository contains the code for the experiments conducted in the following paper: ApproxIFER: ApproxIFER: A Model-Agnostic Approach to Resilient and Robust Prediction Serving Systems 
https://ojs.aaai.org/index.php/AAAI/article/view/20809


Reproducing Results

To reproduce the results presented in the paper, run the scripts that start with "Workstation". For example, Workstation-CIFAR.py returns the accuracy of predictions over the CIFAR dataset. If no additional information is provided, the code only considers the scenario with stragglers and no adversarial servers.

Error-Tolerant Implementations
Scripts with "ERROR" in their name handle scenarios with Byzantine (adversarial) servers. The specific parameters, such as the number of adversarial servers, can be configured within the code. For example, Workstation-MNIST-Error.py implements inference over the MNIST dataset with adversarial servers.

Extensions
Some scripts extend the original paper's work and are still in progress. These include:

Workstation-MNIST-Approximate-error-locator.py: Uses rational functions instead of polynomials to locate errors, reducing redundancy needed to locate errors by a factor of 2.
Workstation_MNIST_Error_Median_Decoder.py: Implements the median decoder for the adversarial setup as a baseline.


AWS Experiments
The following scripts are for running experiments on Amazon AWS:

MNIST_AWS_Latency_Experiment.py: Experiments with stragglers.
MNIST_Latency_Error_revised.py: Experiments with adversarial servers.
