import torch
from torch import nn
from torchvision import datasets
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.is_available()

input_size = 100
#generate the size of random noise
batch_size = 200
#data preprocessing
dataset = datasets.MNIST('data/',download=True)
data = dataset.data.reshape(-1, 1, 28, 28).float()
data = data / (255/2) - 1

class DNet(nn.Module):
    # Discriminator，recognizes an image and returns the probability, the closer to 1, the more likely be an real image
    # input:(batch_size, 1, 28, 28)
    # output:(batch_size, 1)
    def __init__(self):
        super(DNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, 3, 2, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, 2, padding=1)
        self.conv3 = nn.Conv2d(128, 256, 3, 2, padding=1)
        self.conv4 = nn.Conv2d(256, 512, 3, 1, padding=1)
        self.batch_norm1 = torch.nn.BatchNorm2d(128)
        self.batch_norm2 = torch.nn.BatchNorm2d(256)
        self.batch_norm3 = torch.nn.BatchNorm2d(512)
        self.leakyrelu = nn.LeakyReLU()
        self.linear = nn.Linear(8192, 1)

    def forward(self, x):
        x = self.leakyrelu(self.conv1(x))
        x = self.leakyrelu(self.batch_norm1(self.conv2(x)))
        x = self.leakyrelu(self.batch_norm2(self.conv3(x)))
        x = self.leakyrelu(self.batch_norm3(self.conv4(x)))
        x = torch.flatten(x).reshape(-1, 8192)
        x = torch.sigmoid(self.linear(x))
        return x

class GNet(nn.Module):
    # Generator，input random noise and generates images as real as possible
    # input:(batch_size, noise_size)
    # output:(batch_size, 1, 28, 28)
    def __init__(self, input_size):
        super(GNet, self).__init__()
        self.d = 3
        self.linear = nn.Linear(input_size, self.d*self.d*512)
        self.conv_tranpose1 = nn.ConvTranspose2d(512, 256, 5, 2, 1)
        self.conv_tranpose2 = nn.ConvTranspose2d(256, 128, 4, 2, 1)
        self.conv_tranpose3 = nn.ConvTranspose2d(128, 64, 4, 2, 1)
        self.conv_tranpose4 = nn.ConvTranspose2d(64, 1, 3, 1, 1)
        self.batch_norm1 = torch.nn.BatchNorm2d(512)
        self.batch_norm2 = torch.nn.BatchNorm2d(256)
        self.batch_norm3 = torch.nn.BatchNorm2d(128)
        self.batch_norm4 = torch.nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.linear(x).reshape(-1, 512, self.d, self.d)
        x = self.relu(self.batch_norm1(x))
        x = self.conv_tranpose1(x)
        x = self.relu(self.batch_norm2(x))
        x = self.conv_tranpose2(x)
        x = self.relu(self.batch_norm3(x))
        x = self.conv_tranpose3(x)
        x = self.relu(self.batch_norm4(x))
        x = self.tanh(self.conv_tranpose4(x))
        return x

dmodel = DNet().to(device)
gmodel = GNet(input_size).to(device)

#Use BCEloss because MSE is more difficult to convergence than BCE
loss_fun = nn.BCELoss()
goptim = torch.optim.Adam(gmodel.parameters(), lr=0.0001)
doptim = torch.optim.Adam(dmodel.parameters(), lr=0.0001)

# training
dmodel.train()
gmodel.train()
li_gloss = []
li_dloss = []

d_true = torch.ones(batch_size, 1).to(device)
d_fake = torch.zeros(batch_size, 1).to(device)
for epoch in range(80):
    for batch in range(0, 60000, batch_size):
        # real data from MINST
        batch_data = data[batch:batch+batch_size].to(device)
        # fake data generated randomly
        fake_data = torch.randn(batch_size, input_size).to(device)
        # discriminator works for real data first (close to 1 is better)
        output_dtrue = dmodel(batch_data)
        loss_dtrue = loss_fun(output_dtrue, d_true)
        # than for fake data (close to 0 is better)
        output_dfake = dmodel(gmodel(fake_data))
        loss_dfake = loss_fun(output_dfake, d_fake)
        loss_d = loss_dtrue + loss_dfake
        doptim.zero_grad()
        loss_d.backward()
        doptim.step()
        li_dloss.append(loss_d)
        # training generator after discriminator
        # train the generator 3 times every time the discriminator is trained
        for i in range(3):
            output_gtrue = dmodel(gmodel(fake_data))
            # evaluate the generated image from fake data
            loss_g = loss_fun(output_gtrue, d_true)
            # close to 1 is better
            doptim.zero_grad()
            goptim.zero_grad()
            loss_g.backward()
            goptim.step()
            li_gloss.append(loss_g)
        print("epoch:{}, batch:{}, loss_d:{}, loss_g:{}".format(epoch, batch, loss_d, loss_g))
        torch.save(dmodel.state_dict(), "gan_dmodel.mdl")
        torch.save(gmodel.state_dict(), "gan_gmodel.mdl")
        if batch / batch_size % 30 == 0 and batch != 0:
            plt.plot(li_dloss)
            plt.show()
            plt.plot(li_gloss)
            plt.show()

# load existed model
# gmodel.load_state_dict(torch.load("gan_gmodel.mdl"))
# dmodel.load_state_dict(torch.load("gan_dmodel.mdl"))

# test
gmodel.eval()
data_test = torch.randn(100, 100)
result = gmodel(data_test.to(device))
plt.figure(figsize=(10, 50))
for i in range(len(result)):
    ax = plt.subplot(len(result) / 5, 5, i+1)
    plt.imshow((result[i].cpu().data.reshape(28, 28)+1)*255/2)
    # plt.gray()
plt.savefig('./result.png')
plt.show()