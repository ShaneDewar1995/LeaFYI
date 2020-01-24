# Leaf features based plant identification!

![logo](https://github.com/ShaneDewar1995/LeaFYI/blob/master/Images/ic_launcher_leaf_round.png?raw=true)

This project documents the design and implementation of an automated plant species identification software using a deep learning Convolution Neural Network (CNN). 
Plant identification proves to be a very complex and challenging task as there are an estimated 391 000 different species of plants, with many of them having very similar leaf shapes and patterns. It has however, been discovered that each species of trees leaves are unique; like a human fingerprint; and can therefore be used to conclusively categorize plants by their leaves. 
Following the initial research, A Wide Residual Neural Network (ResNet) was determined as the most effective solution. 
Testing shows that the implemented solution can classify 76 different species of plants with an accuracy of 98.8%.Data augmentation and transfer learning played a key role in successfully training the network on the small dataset.

# Mobile Application
The android phone app allows users identify plants on-the-go by simply snapping a picture with their camera. The app uploads the image to the flask server; which passes the image through the neural network; and returns the classification as a JSON Response. Shown below

![server connection](https://github.com/ShaneDewar1995/LeaFYI/blob/master/Images/server.png?raw=true)

Screenshots of the phone app are shown below:

![home screenshot](https://github.com/ShaneDewar1995/LeaFYI/blob/master/Images/home.jpg?raw=true)

![id screenshot](https://github.com/ShaneDewar1995/LeaFYI/blob/master/Images/id.jpg?raw=true)

# Neural Network Training

The CNN was trained in python using pytorch, and produced a 98.8% testing accuracy. This result is competitive among the best results obtained by other research papers in the field.

The training loss and accuracy results are shown below respectively.

![loss](https://github.com/ShaneDewar1995/LeaFYI/blob/master/Images/Loss%20Graph.png?raw=true)

![accuracy](https://github.com/ShaneDewar1995/LeaFYI/blob/master/Images/Accuracy%20graph.png?raw=true)

### Further reading

If you are interested in reading the full implementation of the project , feel free to read my report in the docs folder. 
For any questions contact me: shanerdewar@gmail.com

