# EyeGaze

Eye-tracking is a method that involves monitoring the position and movement of eyes that can help towards wide variety of applications.
We use L2CS-Net, a novel gaze estimation architecture built on ResNet-50. The model, is validated on the MPII Gaze dataset through 5-fold cross-validation and exhibited consistent reduction in combined loss during training.
The performance of the model is demonstrated for an application of profiling the attention of the user and monitoring the off-screen glances of the user while watching a video. 

## *Pipeline of the Model*
![Image](PIPELINE.png)

Data Pre-processing
Pre-processing is vital for standardization and normalization of data, ensuring efficient feature learning.
Data is been preprocessed as given in [here](https://phi-ai.buaa.edu.cn/Gazehub/3D-dataset/).

Data is been preprocessed as given in here https://phi-ai.buaa.edu.cn/Gazehub/

L2CS-Net Architecture

Model Backbone: L2CS-Net utilizes ResNet-50 as a backbone for gaze estimation.
Distinctive Features: Optimized gaze estimation with distinctive features integrated into the ResNet-50 architecture.
Dual Fully-Connected Layers: Unlike traditional approaches, L2CS-Net uses separate fully-connected layers followed by softmax layers for yaw and pitch angles.
Loss Functions: Mean squared error and cross entropy losses are applied to fine-tune the network weights for accurate gaze prediction.
Classification and Regression Strategy: Utilizes a combined strategy with SoftMax activation, categorical bins for gaze direction, and continuous bins for precise numerical values.
Expectation Calculation: Calculates the expectation of the probability distribution for fine-grained gaze predictions.
Final Loss: Combines cross-entropy loss and mean squared error for comprehensive performance evaluation.
