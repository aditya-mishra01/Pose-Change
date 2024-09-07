import torch
import torch.nn as nn
from losses import get_cycle_gen_loss,get_disc_loss
from imageDataset import ImageAndGT
from show_tensor_image import show_tensor_images
from CycleGan import Generator,Discriminator
from torch.utils.data import DataLoader,Dataset


adv_criterion = nn.MSELoss()
recon_criterion = nn.L1Loss()

n_epochs = 100
dim_A = 3
dim_B = 3
display_step = 50

lr = 0.002 #previous - 0.001
lr_d=0.001 # previous - 0.0006
target_shape = 256
device = 'cuda'

gen = Generator(dim_A, dim_B).to(device)
gen_opt = torch.optim.Adam(gen.parameters(), lr=lr, betas=(0.5, 0.999))
disc = Discriminator(dim_A).to(device)
disc_opt = torch.optim.Adam(disc.parameters(), lr=lr_d, betas=(0.5, 0.999))
# disc_B = Discriminator(dim_B).to(device)
# disc_B_opt = torch.optim.Adam(disc_B.parameters(), lr=lr_d, betas=(0.5, 0.999))

def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m,nn.Linear):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    if isinstance(m, nn.BatchNorm2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
        torch.nn.init.constant_(m.bias, 0)



gen = gen.apply(weights_init)
disc = disc.apply(weights_init)
# disc_B = disc_B.apply(weights_init)
# checkpoint = torch.load("model_checkpoint.pth")

# # Load the generator and its optimizer state
# gen.load_state_dict(checkpoint['gen_state_dict'])
# gen_opt.load_state_dict(checkpoint['gen_opt_state_dict'])

# # Load the discriminator and its optimizer state
# disc.load_state_dict(checkpoint['disc_state_dict'])
# disc_opt.load_state_dict(checkpoint['disc_opt_state_dict'])

data=ImageAndGT("0_img","0_gt",transform=transform)
print(len(data))
A=DataLoader(data,batch_size=6,shuffle=True)
B=DataLoader(data,batch_size=6,shuffle=True)

def train():

    mean_discriminator_loss=0
    mean_generator_loss=0
    cur_step = 0

    for epoch in range(n_epochs):
        for x_data, y_data in zip(A,B):


          real_A,gt_A=x_data
          real_B,gt_B=y_data
            # real_A = nn.functional.interpolate(real_A, size=target_shape)
            # real_B = nn.functional.interpolate(real_B, size=target_shape)

          cur_batch_size = len(real_A)
          real_A = real_A.to(device)
          real_B = real_B.to(device)
          gt_A=gt_A.to(device)
          gt_B=gt_B.to(device)

          disc.zero_grad()
          with torch.no_grad():
              fake_A = gen(real_B,gt_A)
              fake_B=gen(real_A,gt_B)
          disc_loss = get_disc_loss(real_A, fake_A,real_B,fake_B, disc, adv_criterion)
          disc_loss.backward(retain_graph=True)
          disc_opt.step()


          gen_opt.zero_grad()
          gen_loss, fake_A, fake_B = get_gen_loss(
                    real_A, real_B,gt_A,gt_B,gen, disc, adv_criterion, recon_criterion, recon_criterion
            )

          gen_loss.backward()
          gen_opt.step()


          mean_discriminator_loss += disc_loss.item() / (display_step)

          mean_generator_loss += gen_loss.item() / (display_step)


          if cur_step % display_step == 0:
            print(f"Epoch {epoch}: Step {cur_step}: Generator (U-Net) loss: {mean_generator_loss}, Discriminator loss: {mean_discriminator_loss}")
                # show_tensor_images(torch.cat([real_A, real_B]), size=(dim_A, target_shape, target_shape))
                # show_tensor_images(torch.cat([fake_B, fake_A]), size=(dim_B, target_shape, target_shape))
            show_tensor_images(real_A,num_images=3, size=(dim_A, target_shape, target_shape))
            show_tensor_images(real_B,num_images=3 ,size=(dim_A, target_shape, target_shape))
            show_tensor_images(fake_A,num_images=3, size=(dim_A, target_shape, target_shape))
            show_tensor_images(fake_B,num_images=3 ,size=(dim_A, target_shape, target_shape))
            mean_discriminator_loss=0
            mean_generator_loss=0
          cur_step += 1


      # print(f"Epoch {epoch}: Step {cur_step}: Generator (U-Net) loss: {mean_generator_loss}, Discriminator loss: {mean_discriminator_loss}")
train()

print("Save Model:")
if(input().lower()=="y"):
    
    torch.save({
        'gen_AB': gen_AB.state_dict(),
        'gen_BA': gen_BA.state_dict(),
        'gen_opt': gen_opt.state_dict(),
        'disc_A': disc_A.state_dict(),
        'disc_A_opt': disc_A_opt.state_dict(),
        'disc_B': disc_B.state_dict(),
        'disc_B_opt': disc_B_opt.state_dict()
            }, f"cycleGAN.pth")