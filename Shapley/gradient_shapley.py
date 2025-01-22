import os
import torch
import random
import pickle
import medmnist
import numpy as np
from model import Net
import torch.nn as nn
from medmnist import INFO
import torch.optim as optim
from models import ResNet18
import torchvision.transforms as transforms
from index_data_loader import IndexDataLoader
from sklearn.metrics import accuracy_score
from torchvision.models import resnet18

def seed_torch(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def replace_batchnorm_with_instancenorm(model):
  for name, module in model.named_children():
      if isinstance(module, nn.BatchNorm2d):
          # Replace BatchNorm2d with InstanceNorm2d
          num_features = module.num_features
          instance_norm = nn.InstanceNorm2d(num_features, affine=True)
          setattr(model, name, instance_norm)
      else:
          # Recursively apply to child modules
          replace_batchnorm_with_instancenorm(module)

def get_pretrained_model(model_name, num_classes, n_channels = None):
    model = models[model_name](pretrained=True)

    if n_channels is not None:
        model.conv1 = nn.Conv2d(n_channels, model.conv1.out_channels, kernel_size=model.conv1.kernel_size, stride=model.conv1.stride, padding=model.conv1.padding, bias=False)

    # Freeze all layers
    for param in model.parameters():
        param.requires_grad = False

    # Replace the final fully connected layer
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes) 

    # Only the parameters of the final layer will be updated
    for param in model.fc.parameters():
        param.requires_grad = True

    return model


def getACC(y_true, y_score, task, threshold=0.5):
    """Accuracy metric.
    :param y_true: the ground truth labels, shape: (n_samples, n_labels) or (n_samples,) if n_labels==1
    :param y_score: the predicted score of each class,
    shape: (n_samples, n_labels) or (n_samples, n_classes) or (n_samples,) if n_labels==1 or n_classes==1
    :param task: the task of current dataset
    :param threshold: the threshold for multilabel and binary-class tasks
    """ 
    y_true = y_true.squeeze(1)
    if y_score.shape[0] > 1:
        y_score = y_score.squeeze()

    if task == "multi-label, binary-class":
        y_pre = y_score > threshold
        acc = 0
        for label in range(y_true.shape[1]):
            label_acc = accuracy_score(y_true[:, label], y_pre[:, label])
            acc += label_acc
        ret = acc / y_true.shape[1]
    elif task == "binary-class":
        if y_score.ndim == 2:
            y_score = y_score[:, -1]
        else:
            assert y_score.ndim == 1
        ret = accuracy_score(y_true, y_score > threshold)
    else:
        ret = accuracy_score(y_true, np.argmax(y_score, axis=-1))

    return ret

def get_model(n_channels, n_classes, device, custom_model = None):
    model = None
    if custom_model is None:
        model = ResNet18(in_channels=n_channels, num_classes=n_classes)
        #Net(in_channels=n_channels, num_classes=n_classes)
    else:
        model = get_pretrained_model(custom_model, n_classes, n_channels)
    
    model.to(device)
    return model

def train(loader, model, task, device, epoch):
    # define loss function and optimizer
    if task == "multi-label, binary-class":
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # train

    for epoch in range(epoch):
        model.train()
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            # forward + backward + optimize
            optimizer.zero_grad()
            outputs = model(inputs)

            if task == 'multi-label, binary-class':
                targets = targets.to(torch.float32)
                loss = criterion(outputs, targets)
            else:
                targets = targets.squeeze(1).long()
                loss = criterion(outputs, targets)

            loss.backward()
            optimizer.step()
    
    return model

def test(data_loader, model, task, data_flag, device):
    model.eval()
    y_true = torch.tensor([]).to(device)
    y_score = torch.tensor([]).to(device)

    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)

            if task == 'multi-label, binary-class':
                outputs = outputs.softmax(dim=-1)
            else:
                targets = targets.squeeze(1).long()
                outputs = outputs.softmax(dim=-1)
                targets = targets.float().resize_(len(targets), 1)

            y_score = torch.cat((y_score, outputs), 0)
            y_true = torch.cat((y_true, targets), 0)

        y_score = y_score.detach().cpu().numpy()
        y_true = y_true.detach().cpu().numpy()

        accuracy = getACC(y_true, y_score, task)
        return accuracy

def main(data_flag, custom_model, nbr_iter):

    info = INFO[data_flag]
    task = info['task']
    n_channels = info['n_channels']
    n_samples = info['n_samples']['train']
    n_classes = len(info['label'])

    seed_torch(42)

    DataClass = getattr(medmnist, info['python_class'])

    # preprocessing
    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[.5], std=[.5])
    ])

    train_dataset = DataClass(split='train', transform=data_transform, download=True)
    device = torch.device('cuda')

    # Truncated Monte Carlo Shapley
    t = 0
    v_t = []
    pi_t = []
    # Initialize
    v_t.append(np.zeros(n_samples))
    shapley_values = np.zeros(n_samples)
    pi_t.append(np.array(list(range(n_samples)))) # pi_0
    model = get_model(n_channels, n_classes, device, custom_model=custom_model)

    while t < nbr_iter:

        print("t:", t)  
        t = t + 1
        pi_t.append(list(np.random.permutation(n_samples)))
        pi_t[t].insert(0, -1) # dummy value
        v_t.append(np.zeros(n_samples))
        v_t[t][0] = 0 # v_0_t = V(0, A)
        
        for j in range(1, n_samples):
            
            # DİKKAT - PRETRAINED MODELLER TEK INPUT İLE ÇALIŞMIYOR
            indexes = None
            indexes = pi_t[t][j:j+1]
            if custom_model is not None:
                indexes.extend(pi_t[t][j:j+1])
            train_index_loader = IndexDataLoader(train_dataset, indexes, 128)
            
            model = train(train_index_loader, model, task, device, 1)
            v_t[t][j] = test(train_index_loader, model, task, data_flag, device)

            if t > 1:
                shapley_values[pi_t[t][j]] = (((t - 1) / t) * shapley_values[pi_t[t - 1][j]]) + ((1/t) * (v_t[t][j] - v_t[t][j - 1])) 
            else:
                shapley_values[pi_t[t][j]] = ((1/t) * (v_t[t][j] - v_t[t][j - 1])) 

    with open(f"shapley_files/tmc_v2_{data_flag}_{nbr_iter}_times.pkl", "wb") as f:
        pickle.dump(shapley_values, f)

if __name__ == '__main__':
    models = {'resnet18': resnet18}
    dataset_list = ['pathmnist', 'dermamnist', 'octmnist', 'pneumoniamnist', 'retinamnist', 
                    'breastmnist', 'bloodmnist', 'tissuemnist', 'organamnist', 'organcmnist', 'organsmnist']
    for data_flag in dataset_list:
        print(data_flag)
        main(data_flag=data_flag, custom_model='resnet34', nbr_iter=10)