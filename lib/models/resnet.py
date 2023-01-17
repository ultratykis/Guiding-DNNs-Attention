from tabnanny import check
import torch.optim as optim
import torch.nn as nn
from torchvision import transforms, datasets, models
import torch
import os
from tqdm import tqdm

data_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
])


def train(dataset_path, network_path):
    train_dataset = datasets.ImageFolder(
        root=dataset_path+'train', transform=data_transform)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=256, shuffle=True, num_workers=4)
    print(train_dataset.class_to_idx)

    net = models.resnet18(num_classes=2)
    net.train()
    net.cuda()

    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = optim.Adam(net.parameters(), lr=0.1)
    best_loss = 0.0

    for epoch in range(50):
        running_loss = 0.0
        net.train()

        for data in train_loader:
            inputs, labels = data

            inputs = inputs.cuda()
            labels = labels.cuda()

            optimizer.zero_grad()
            outputs = net(inputs)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        if best_loss == 0.0:
            best_loss = running_loss
            print('epoch:', epoch, "no improvement, best loss: {}, running loss: {}".format(
                best_loss, running_loss))
        elif best_loss > running_loss:
            best_loss = running_loss
            # Save the model as checkpoint.
            print('epoch:', epoch, "update best loss: {}, Saving model...".format(
                best_loss))
            if not os.path.exists(network_path):
                os.mkdir(network_path)
            torch.save(
                {
                    'model_state_dict': net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, network_path + 'resnet18.pth')
            print("Saved best model to {}".format(
                network_path + 'resnet18.pth'))
        else:
            print('epoch:', epoch, "no improvement, running loss: {}, best loss: {}".format(
                running_loss, best_loss))

        running_loss = 0.0
    print('Train finished!')


def test(dataset_path, net_path, which_set):
    dataset_path = dataset_path + which_set

    test_dataset = datasets.ImageFolder(
        root=dataset_path, transform=data_transform)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=128, shuffle=True, num_workers=4)

    net = models.resnet18(num_classes=2)
    net.cuda()
    net.load_state_dict(torch.load(
        net_path + 'resnet18.pth')['model_state_dict'])

    net.eval()

    correct = 0
    total = 0

    class_correct = list(0. for i in range(2))
    class_total = list(0. for i in range(2))

    with torch.no_grad():
        for data in tqdm(test_loader, desc='Testing...'):
            images, labels = data

            images = images.cuda()
            labels = labels.cuda()

            outputs = net(images)

            batch_total, batch_correct, batch_class_total, batch_class_correct = get_accuracy(
                outputs, labels)

            total += batch_total
            correct += batch_correct
            for i in range(2):
                class_total[i] += batch_class_total[i]
                class_correct[i] += batch_class_correct[i]

    print('Accuracy of the network on the {} test images: {} %, correct: {}'.format(
        total, correct / total, correct))
    for i in range(2):
        print('Accuracy of {}: {} %, correct num: {}, total: {}'.format(
            i, class_correct[i] / class_total[i], class_correct[i], class_total[i]))


def get_accuracy(outputs, labels):
    _, predicted = torch.max(outputs, 1)
    batch_total = labels.size(0)
    batch_correct = (predicted == labels).sum().item()

    batch_class_correct = list(0. for i in range(2))
    batch_class_total = list(0. for i in range(2))

    c = (predicted == labels).squeeze()
    for i in range(outputs.size(0)):
        label = labels[i]
        batch_class_total[label] += 1
        batch_class_correct[label] += c[i].item()

    return batch_total, batch_correct, batch_class_total, batch_class_correct


if __name__ == "__main__":
    train("./datasets/coco-2017/", "./checkpoints/")
