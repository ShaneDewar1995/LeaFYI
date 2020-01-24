import torch
import torch.nn as nn
from PIL import Image
import torchvision
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import Constants
import time
import os
import random
from Constants import connection
from decimal import Decimal, ROUND_HALF_DOWN
from ConvNet import WideResnet
from torchvision import datasets, transforms

torch.cuda.current_device()


class LeafsnapData:
    def __init__(self):
        # initializing the data directories
        self.data_dir_train = Constants.DATA_DIR_TRAIN
        self.data_dir_test = Constants.DATA_DIR_TEST

    def load_split_train_test(self):
        train_transforms = transforms.Compose([transforms.Resize((224, 224)),
                                               transforms.RandomHorizontalFlip(),
                                               transforms.RandomVerticalFlip(),
                                               transforms.RandomRotation(30),
                                               transforms.ColorJitter(brightness=0.5, contrast=0.5),
                                               transforms.ToTensor(),
                                               transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                               ])
        test_transforms = transforms.Compose([transforms.Resize((224, 224)),
                                              transforms.ToTensor(),
                                              transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                              ])
        train_data = datasets.ImageFolder(self.data_dir_train,
                                          transform=train_transforms)
        test_data = datasets.ImageFolder(self.data_dir_test,
                                         transform=test_transforms)

        trainloader = torch.utils.data.DataLoader(train_data, batch_size=8, shuffle=True)
        testloader = torch.utils.data.DataLoader(test_data, batch_size=8, shuffle=True)
        return trainloader, testloader


# Training the Network
class Train:
    def __init__(self, net, device, training, testing):
        print("initializing training")
        self.net = net
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(net.parameters(), lr=0.001)
        self.trainingset = training
        self.testingset = testing

    def training(self):
        losses = []
        learning_rate = []
        print("Training now")
        start = time.time()
        scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=4, gamma=0.5)
        for epoch in range(10):  # loop over the data set multiple times 10 = num_epochs

            scheduler.step(epoch)
            learning_rate.append(scheduler.get_lr())
            running_loss = 0.0
            count = 0

            for i, data in enumerate(self.trainingset, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data[0].to(self.device), data[1].to(self.device)

                # zero the parameter gradients
                self.optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self.net(inputs)
                loss = self.criterion(outputs, labels)

                loss.backward()
                self.optimizer.step()

                # Accumulating the loss value
                running_loss += loss.item()
                count += 1
            torch.save(self.net.state_dict(), Constants.MODEL_STORE_PATH + 'Plant_model.ckpt')
            print('[%d] loss: %.3f' %
                  (epoch + 1, running_loss / count))
            losses.append(running_loss / count)
            running_loss = 0.0

        print('Finished Training')
        end = time.time()
        print("training time: ", (end - start))
        print(learning_rate)
        # Saving the trained model
        torch.save(self.net.state_dict(), Constants.MODEL_STORE_PATH + 'Plant_model.ckpt')

        # return the loss vector
        return losses


class PlotStats:
    def __init__(self, losses):
        self.losses = losses

    def lossplot(self):
        plt.plot(self.losses)
        plt.ylabel("loss")
        plt.xlabel("epoch")
        plt.show()


# Testing the network
class Test:
    def loadNetwork(self, net):
        net.load_state_dict(torch.load(Constants.MODEL_STORE_PATH + 'Plant_model.ckpt'))
        net.eval()

    def testdisplay(self, net, testing):
        testingset = testing
        dataiter = iter(testingset)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(device)
        images, labels = dataiter.next()
        # print images
        img = torchvision.utils.make_grid(images)
        img = img / 2 + 0.5  # unnormalize
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.show()
        print('GroundTruth: ', ' '.join('%5s' % testingset.dataset.classes[labels[j]] for j in range(8)))
        outputs = net(images.to(device))
        _, predicted = torch.max(outputs, 1)
        print('Predicted: ', ' '.join('%5s' % testingset.dataset.classes[predicted[j]] for j in range(8)))

        class_correct = list(0. for i in range(76))
        class_total = list(0. for i in range(76))
        correct = 0
        total = 0
        with torch.no_grad():
            for data in testingset:
                images, labels = data[0].to(device), data[1].to(device)
                outputs = net(images)
                _, predicted = torch.max(outputs, 1)
                c = (predicted == labels).squeeze()
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
                for i in range(len(data[1])):
                    label = labels[i]
                    class_correct[label] += c[i].item()
                    class_total[label] += 1
        for i in range(76):
            print('Accuracy of %5s : %2d %%' % (
                testingset.dataset.classes[i], 100 * class_correct[i] / class_total[i]))
        print("Total correct percent: ", (correct / total) * 100)


class ClassifyImage:
    def __init__(self):
        obj = WideResnet()
        self.net = obj.MyWideResnet()
        self.net.load_state_dict(torch.load(Constants.MODEL_STORE_PATH + 'Plant_model.ckpt'))
        self.net.eval()

    def my_image(self, direct):
        classification, name, sci_name, edible, url, probability = "", "", "", "", "", ""
        try:
            real = Image.open(direct)
            x = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
            inp = x(real).unsqueeze_(0)
            outputs = self.net(inp)
            _, predicted = torch.max(outputs, 1)

            # get Confidence/probability
            sm = torch.nn.Softmax(dim=1)
            prob = sm(outputs)
            probability = Decimal(prob[0][predicted.item()].item() * 100)
            probability = Decimal(probability.quantize(Decimal('.01'), rounding=ROUND_HALF_DOWN))
            probability = str(probability)

            # Fetch DB values for plant
            mycursor, mydb = connection()
            print(predicted.item())
            sql = """SELECT * FROM `PlantSpecies` WHERE `PlantID` = %s"""
            mycursor.execute(sql, (predicted.item() + 1,))
            myresult = mycursor.fetchall()
            for row in myresult:
                classification = row[0]
                name = row[1]
                sci_name = row[2]
                edible = row[3]
                url = row[4]

        except Exception as e:
            print("failed to classify image")
            print(e)
        return classification, name, sci_name, edible, url, probability


class UITest:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(self.device)
        obj = WideResnet()
        self.net = obj.MyWideResnet()
        self.net.load_state_dict(torch.load(Constants.MODEL_STORE_PATH + 'Plant_model.ckpt'))
        self.net.eval()
        data = LeafsnapData()
        self.trainloader, self.testloader = data.load_split_train_test()

    def single(self):
		# use cpu for a single image test
        self.net.to("cpu")
        dataiter = iter(self.testloader)
        images, labels = dataiter.next()
        groundtruth = self.testloader.dataset.classes[labels[0]]
        path = Constants.DATA_DIR + groundtruth
        random_filename = random.choice([
            x for x in os.listdir(path)
            if os.path.isfile(os.path.join(path, x))
        ])
        filepath = path + '/' + random_filename
        print(filepath)
        outputs = self.net(images)
        _, predicted = torch.max(outputs, 1)
        predicted = self.testloader.dataset.classes[predicted[0]]
        return groundtruth, predicted, filepath

	# determine the overall test accuracy
    def network_accuracy(self):

        class_correct = list(0. for i in range(76))
        class_total = list(0. for i in range(76))
        correct = 0
        total = 0
        self.net.to(self.device)
        with torch.no_grad():
            for data in self.testloader:
                images, labels = data[0].to(self.device), data[1].to(self.device)
                outputs = self.net(images)
                _, predicted = torch.max(outputs, 1)
                c = (predicted == labels).squeeze()
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
                for i in range(len(data[1])):
                    label = labels[i]
                    class_correct[label] += c[i].item()
                    class_total[label] += 1
        for i in range(76):
            print('Accuracy of %5s : %2d %%' % (
                self.testloader.dataset.classes[i], 100 * class_correct[i] / class_total[i]))
        print("Total correct percent: ", (correct / total) * 100)
        classes_accuracy = "Accuracy of: \n"
        for i in range(76):
            temp = '   ' + self.testloader.dataset.classes[i] + ' : '\
                   + str(round(100 * class_correct[i] / class_total[i], 2))
            classes_accuracy = classes_accuracy + temp + '%\n'

        total_perc = str(round((correct / total) * 100, 2)) + "%"

        return classes_accuracy, total_perc

