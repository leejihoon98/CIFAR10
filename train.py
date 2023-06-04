import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import timm
import random
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from albumentations.core.transforms_interface import ImageOnlyTransform
from tqdm.auto import tqdm
import warnings
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)


model_num = 6 
total_epoch = 100 
lr = 0.0001 



# class AugMix(ImageOnlyTransform):

#     def __init__(self, width=2, depth=2, alpha=0.5, augmentations=[A.HorizontalFlip()], always_apply=False, p=0.5):
#         super(AugMix, self).__init__(always_apply, p)
#         self.width = width
#         self.depth = depth
#         self.alpha = alpha
#         self.augmentations = augmentations
#         self.ws = np.float32(np.random.dirichlet([self.alpha] * self.width))
#         self.m = np.float32(np.random.beta(self.alpha, self.alpha))

#     def apply_op(self, image, op):
#         image = op(image=image)["image"]
#         return image

#     def apply(self, img, **params):
#         mix = np.zeros_like(img)
#         for i in range(self.width):
#             image_aug = img.copy()

#             for _ in range(self.depth):
#                 op = np.random.choice(self.augmentations)
#                 image_aug = self.apply_op(image_aug, op)

#             mix = np.add(mix, self.ws[i] * image_aug, out=mix, casting="unsafe")

#         mixed = (1 - self.m) * img + self.m * mix
#         if img.dtype in ["uint8", "uint16", "uint32", "uint64"]:
#             mixed = np.clip((mixed), 0, 255).astype(np.uint8)
#         return mixed

#     def get_transform_init_args_names(self):
#         return ("width", "depth", "alpha")

# augs = [A.HorizontalFlip(always_apply=True),
#         A.Blur(always_apply=True),
#         A.OneOf(
#         [A.ShiftScaleRotate(always_apply=True),
#         A.GaussNoise(always_apply=True)]
#         ),
#         A.CoarseDropout(always_apply=True)
#         ]


# Load CIFAR-10 dataset
# trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
#                                         download=True, transform=transform_train)
# trainloader = torch.utils.data.DataLoader(trainset, batch_size=128,
#                                           shuffle=True, num_workers=16)

# testset = torchvision.datasets.CIFAR10(root='./data', train=False,
#                                        download=True, transform=transform_test)
# testloader = torch.utils.data.DataLoader(testset, batch_size=128,
#                                          shuffle=False, num_workers=16)

for s in range(model_num):
   
    seed_number = s
    random.seed(seed_number)
    np.random.seed(seed_number)
    torch.manual_seed(seed_number)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    # transform_train = A.Compose([
    # A.Resize(256,256),
    # A.CenterCrop(224,224),
    # AugMix(width=3, depth=2, alpha=.2, p=1., augmentations=augs),
    # A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1),
    # A.RandomGamma(gamma_limit=(90, 110)),
    # A.ShiftScaleRotate(scale_limit=0.1, rotate_limit=10),
    # A.Transpose(),
    # A.RandomRotate90(),
    # A.OneOf([A.NoOp(), A.MultiplicativeNoise(), A.GaussNoise(), A.ISONoise()]),
    # A.OneOf([
    #             A.NoOp(p=0.8),
    #             A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=10),
    #             A.RGBShift(r_shift_limit=10, g_shift_limit=10, b_shift_limit=10)
    #         ],
    #         p=0.2,
    #         ),
    #     A.OneOf([A.ElasticTransform(), A.GridDistortion(), A.NoOp()]),
    #     A.Normalize( mean = (0.4914, 0.4822, 0.4465),std = (0.2470, 0.2435, 0.2616),p =1.0),
    #     ToTensorV2(),
    #     ])
        
    # transform_test = A.Compose([
    #     A.Resize(256,256),
    #     A.CenterCrop(224,224),
    #     A.Normalize( mean = (0.4914, 0.4822, 0.4465),std = (0.2470, 0.2435, 0.2616),p =1.0),
    #     ToTensorV2()
    #     ])
        
    # class Cifar10SearchDataset(torchvision.datasets.CIFAR10):
    #     def __init__(self, root="./data/cifar10", train=True, download=True, transform=None):
    #         super().__init__(root=root, train=train, download=download, transform=transform)

    #     def __getitem__(self, index):
    #         image, label = self.data[index], self.targets[index]

    #         if self.transform is not None:
    #             transformed = self.transform(image=image)
    #             image = transformed["image"]

    #         return image, label

    # trainset = Cifar10SearchDataset(root='./data/cifar10', transform=transform_train)
    # trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=16)

    transform_train = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.AugMix(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    transform_test = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])


    
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=16, shuffle=True, num_workers=16)


    
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=16)
   
    
    model = timm.create_model('resnet18', pretrained=True, num_classes=10)
    model = model.to(device)  

    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    def train():
        model.train()
        running_loss = 0.0
        
        for i, data in tqdm(enumerate(trainloader, 0)):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)  
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % 100 == 99:
                print('[%d,%5d, %5d] loss: %.3f' % (s,epoch + 1, i + 1, running_loss / 100))
                running_loss = 0.0   
                
    def test():
        model.eval()
        
        
        correct = 0
        total = 0
        with torch.no_grad():
            for data in tqdm(testloader):
                images, labels = data
                images, labels = images.to(device), labels.to(device)  
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print('Accuracy of the network on the 10000 test images: %f %%' % (100 * correct / total))

    
    for epoch in range(total_epoch):
        train()
        test()
        scheduler.step()

    print('Finished Training')

    
    PATH = './resnet18_cifar10_8_%f_%d.pth' % (lr, seed_number)
    torch.save(model.state_dict(), PATH)