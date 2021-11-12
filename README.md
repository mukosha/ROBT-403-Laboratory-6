# ROBT-403-Laboratory-6
ROBT-403-Laboratory-6

# Lab 6
### Initial Conditions:
Initially, the error was as 37.02%:
![image](https://user-images.githubusercontent.com/47817099/141448985-8e928868-4cb4-4026-97bb-fe2f9c780879.png)


### After using the following model:
```script
clear
data_size = 5000; %???
[XTrain, YTrain] = create_dataset_fk(data_size);
% q = [q1, q2, q3, q4, q5];
% 
%  nFeatures = 20; 
%  nExamples = 10000;
%  
%  
%  nOutputs = 1; % this example is for setting up a regression problem
%  
%  x = rand(nExamples,nFeatures); 
%  t = rand(nExamples, nOutputs);

XTrain = reshape(XTrain', [1, 1, size(XTrain,2),size(XTrain,1)]);

nFeatures = 5;
numClasses = 3;

%You can learn about training NN by looking the following links
%https://www.mathworks.com/help/deeplearning/ref/trainingoptions.html
%https://www.mathworks.com/help/deeplearning/ref/trainnetwork.html
%https://www.mathworks.com/help/deeplearning/ug/list-of-deep-learning-layers.html

layers = [ ...
     imageInputLayer([1 1 nFeatures]);
    
    fullyConnectedLayer(512)
    fullyConnectedLayer(256)
    tanhLayer()
%  add more hidden layers to reduce the total mean error VARIABLE 'a' in the
%  script 'demo_fk.m'
%  you can change reluLayer to other activation functions
    fullyConnectedLayer(numClasses)  
    regressionLayer
    ]

maxEpochs = 10; %24
miniBatchSize = 100;

options = trainingOptions('adam', ...
    'ExecutionEnvironment','cpu', ...
    'GradientThreshold',1, ...
    'MaxEpochs',maxEpochs, ...
    'MiniBatchSize',miniBatchSize, ...
    'SequenceLength','longest', ...
    'Shuffle','never', ...
    'Verbose',0, ...
    'Plots','training-progress');

net_fk = trainNetwork(XTrain, YTrain,layers,options);

save net_fk
```
The error was equal to 12.08%:

![image](https://user-images.githubusercontent.com/47817099/141449170-8fe7d338-f34c-40a6-b35d-28094359c117.png)



### After changing a certain amount of the Relu Layers to the LeakyReluLayers:
After the change of the relu layers of 2048-1024-512-256-128  to the leakyReluLayers the error has changed to the 6.11%
The maxEpochs value was equal to the value of 25, and the miniBatchSize was equal to the value of 250.
![image](https://user-images.githubusercontent.com/47817099/141447928-8dbba7e8-078f-4f12-a7f7-b77e56100ebf.png)

The training of the NN:

![image](https://user-images.githubusercontent.com/47817099/141448505-dc64470e-4b33-4335-9266-26c32cf82e5c.png)

### MATLAB Code:
```script
data_size = 2000;
[XTrain, YTrain] = create_dataset_fk(data_size);

XTrain = reshape(XTrain', [1, 1, size(XTrain,2),size(XTrain,1)]);


nFeatures = 5;
numClasses = 3;

layers = [ ...
    imageInputLayer([1 1 nFeatures])
    fullyConnectedLayer(2048)
    leakyReluLayer
    fullyConnectedLayer(1024)
    leakyReluLayer
    fullyConnectedLayer(512)
    leakyReluLayer
    fullyConnectedLayer(256)
    leakyReluLayer
    fullyConnectedLayer(128)
    reluLayer
    fullyConnectedLayer(numClasses)  
    regressionLayer  
    ]

maxEpochs = 25;
miniBatchSize = 250;

options = trainingOptions('adam', ...
    'ExecutionEnvironment','cpu', ...
    'GradientThreshold',1, ...
    'MaxEpochs',maxEpochs, ...
    'MiniBatchSize',miniBatchSize, ...
    'SequenceLength','longest', ...
    'Shuffle','never', ...
    'Verbose',0, ...
    'Plots','training-progress');


net_fk = trainNetwork(XTrain, YTrain,layers,options);


save net_fk
```
