import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from torchvision import datasets
import torchvision.transforms as T
from torchvision.utils import save_image


def main():
    latent_dim = 100

    # Generator
    class Generator(nn.Module):
        def __init__(self):
            super().__init__()
            
            def block(input_dim, output_dim, normalize=True):
                layers = [nn.Linear(input_dim, output_dim)]
                
                if normalize:
                    layers.append(nn.BatchNorm1d(output_dim, 0.8))
                layers.append(nn.ReLU())
                
                return layers
        
            self.model = nn.Sequential(
                *block(latent_dim, 128, normalize=True),
                *block(128,256),
                *block(256,512),
                *block(512, 1024),
                nn.Linear(1024, 1 * 32 * 32),
                nn.Tanh()
            )
            
        def forward(self, z):
            # z는 noise vector
            img = self.model(z)
            img = img.view(img.size(0), 1, 32, 32)
            return img
        
        
    class Discriminator(nn.Module):
        def __init__(self):
            super().__init__()
            
            self.model = nn.Sequential(
                nn.Linear(1*32*32, 512),
                nn.ReLU(),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Linear(256,1),
                nn.Sigmoid() # 확률값으로 내보내기 위해 -> SOFTMAX는 안되나?
            )
            
        def forward(self, img):
            flatten = img.view(img.size(0), -1)
            output = self.model(flatten)
            return output
        
        
    transform = T.Compose([
        T.Resize(32),
        T.ToTensor(),
        T.Normalize([0.5], [0.5]),
    ])

    train_dataset = datasets.MNIST(root="./dataset", train=True, download=True, transform=transform)
    train_dataset = Subset(train_dataset, indices=range(200))
    dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)

    generator = Generator()
    discriminator = Discriminator()

    generator.cuda()
    discriminator.cuda()

    loss_func = nn.BCELoss()

    lr = 1e-3

    optim_g, optim_d = torch.optim.AdamW(generator.parameters(), lr= lr), torch.optim.AdamW(discriminator.parameters(), lr = lr)


    n_epoch = 200
    sampler_interval = 2000

    for epoch in range(n_epoch):
        for idx, (img, label) in enumerate(dataloader):
            real_label = torch.cuda.FloatTensor(img.size(0), 1).fill_(1.0)
            fake_label= torch.cuda.FloatTensor(img.size(0), 1).fill_(0.0)
            real_img = img.cuda()
            
            # noise sampling
            z = torch.normal(mean=0, std=1, size=(img.shape[0], latent_dim)).cuda()

            # generator 학습
            optim_g.zero_grad() # 초기화
            generated_img = generator(z).detach()
            g_loss = loss_func(discriminator(generated_img), real_label)
            g_loss.backward()
            optim_g.step()
            
            # discriminator 학습
            optim_d.zero_grad()
            real_loss = loss_func(discriminator(real_img), real_label)
            fake_loss = loss_func(discriminator(generated_img), fake_label)
            # 어떤 쪽에 무게를 더 둘 것인지
            # 일단은 0.5 | 0.5 로 함
            d_loss = (real_loss + fake_loss) / 2
            d_loss.backward()
            optim_d.step()
            
            # discriminator 학습

            optim_d.zero_grad() # 초기화
            real_loss = loss_func(discriminator(real_img), real_label)
            fake_loss = loss_func(discriminator(generated_img), fake_label) # .detach() 추가
            d_loss = real_loss + fake_loss ** 2
            d_loss.backward()
            optim_d.step()
            
            if epoch % 20 == 0:
                save_image(generated_img[:25], f"{idx}.png", nrow=5, normalize=True)
            
        print(f"Epoch = {epoch}/{n_epoch} | D loss: {d_loss.item():.6f} | G loss: {g_loss.item():.6f}")
        
if __name__=="__main__":
    main()

