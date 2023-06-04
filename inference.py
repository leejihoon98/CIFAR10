import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import timm
from tqdm.auto import tqdm

model_num = 6
lr = 0.0001


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


transform_test = transforms.Compose([
    transforms.Resize(256),
    transforms.TenCrop(224),
    transforms.Lambda(lambda crops: torch.stack([transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(transforms.ToTensor()(crop)) for crop in crops]))
])

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=16)


models = []
for i in range(model_num):
    
    model = timm.create_model('resnet18', num_classes=10)
    model.load_state_dict(torch.load(f"resnet18_cifar10_8_%f_%d.pth" % (lr, i)))  
    model.eval()  
    model = model.to(device)  
    models.append(model)


correct = 0
total = 0
with torch.no_grad():
    for data in tqdm(testloader):
        images, labels = data
        images, labels = images.to(device), labels.to(device) 
        bs, ncrops, c, h, w = images.size()       
        outputs = torch.zeros(bs, 10).to(device)  
        for model in models:
            model_output = model(images.view(-1, c, h, w))  
            model_output = model_output.view(bs, ncrops, -1).mean(1) 
            outputs += model_output
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the ensemble on the 10000 test images: %f %%' % (100 * correct / total))