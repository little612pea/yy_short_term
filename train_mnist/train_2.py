#encoding: utf-8
from model import Model
import numpy as np
import os
import torch
from torchvision.datasets import mnist
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from torch.utils.tensorboard import SummaryWriter  # 添加 TensorBoard 相关导入

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 256
    train_dataset = mnist.MNIST(root='./train', train=True, transform=ToTensor())
    test_dataset = mnist.MNIST(root='./test', train=False, transform=ToTensor())
    train_loader = DataLoader(train_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    model = Model().to(device)
    sgd = SGD(model.parameters(), lr=1e-1)
    loss_fn = CrossEntropyLoss()
    all_epoch = 100
    prev_acc = 0
    
    writer = SummaryWriter(log_dir="logs")  # 创建 TensorBoard 的 SummaryWriter 对象
    
    for current_epoch in range(all_epoch):
        model.train()
        for idx, (train_x, train_label) in enumerate(train_loader):
            train_x = train_x.to(device)
            train_label = train_label.to(device)
            sgd.zero_grad()
            predict_y = model(train_x.float())
            loss = loss_fn(predict_y, train_label.long())
            loss.backward()
            sgd.step()
            
            # 使用 add_scalar 记录训练损失
            writer.add_scalar('Loss/Train', loss.item(), current_epoch * len(train_loader) + idx)
        
        all_correct_num = 0
        all_sample_num = 0
        model.eval()
        
        for idx, (test_x, test_label) in enumerate(test_loader):
            test_x = test_x.to(device)
            test_label = test_label.to(device)
            predict_y = model(test_x.float()).detach()
            # detach 方法将 predict_y 从计算图中分离出来，这样就不会影响后面的反向传播
            predict_y = torch.argmax(predict_y, dim=-1)
            # 使用 argmax 方法求出预测的数字，argmax 方法会返回最大值的索引，dim=-1 表示在最后一个维度上求最大值
            # 最后一个维度是数字的维度，即每个数字的概率，前一个维度是 batch 的维度
            current_correct_num = predict_y == test_label
            all_correct_num += np.sum(current_correct_num.to('cpu').numpy(), axis=-1)
            all_sample_num += current_correct_num.shape[0]
            # shape[0] 返回的是矩阵第一维的长度，即有多少个样本
        
        acc = all_correct_num / all_sample_num
        print('accuracy: {:.3f}'.format(acc), flush=True)
        
        # 使用 add_scalar 记录准确率
        writer.add_scalar('Test Accuracy', acc, current_epoch)
        
        if not os.path.isdir("models"):
            os.mkdir("models")
        torch.save(model, 'models/mnist_{:.3f}.pkl'.format(acc))
        
        if np.abs(acc - prev_acc) < 1e-4:
            break
        
        prev_acc = acc
    
    writer.close()  # 关闭 TensorBoard 的 SummaryWriter
    print("Model finished training")
