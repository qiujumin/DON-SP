from net import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path


batch_size = 16
num_epoch = 1000

data_dir = Path('dataset')

tensor_dict_r = {}
tensor_dict_l = {}

for f in data_dir.glob("*_r.pt"):
    tensor_dict_r[f.name] = torch.load(f)

for f in data_dir.glob("*_l.pt"):
    tensor_dict_l[f.name] = torch.load(f)

total = len(wavelengths)

def get_intensity(E):
    return torch.real(E * torch.conj(E))

def generate_unit_vector():
    vec = np.random.randn(3)
    mag = np.linalg.norm(vec)
    unit_vec = vec / mag
    
    return torch.tensor(unit_vec, dtype=torch.float32)

def generate_data(batch_size, n):

    I_list = []
    spectrum_list = []
    S_list = []

    for _ in range(batch_size):

        S = torch.ones(4)
        S[1:] = generate_unit_vector()

        indices = torch.randperm(total)[:n]
        weights = torch.rand(n)

        spectrum = torch.zeros(total)
        spectrum[indices] = weights

        samples1_r = 0
        samples1_l = 0
        for wavelength, w in zip(wavelengths[indices], weights):

            fname = f"{wavelength.item():.1f}_r_r.pt"
            samples1_r += tensor_dict_r[fname] * w

            fname = f"{wavelength.item():.1f}_r_l.pt"
            samples1_l += tensor_dict_r[fname] * w

        samples2_l = 0
        samples2_r = 0
        for wavelength, w in zip(wavelengths[indices], weights):

            fname = f"{wavelength.item():.1f}_l_l.pt"
            samples2_l += tensor_dict_l[fname] * w

            fname = f"{wavelength.item():.1f}_l_r.pt"
            samples2_r += tensor_dict_l[fname] * w

        delta = -torch.arctan2(S[2], S[1])

        Er = (samples1_r+samples2_r) * torch.sqrt((S[0] + S[3]) / 2)

        El = (
            samples1_l+samples2_l
            * torch.sqrt((S[0] - S[3]) / 2)
            * torch.exp(1j * delta)
        )

        I = get_intensity(Er) + get_intensity(El)

        I_list.append(I)
        spectrum_list.append(spectrum)
        S_list.append(S)

    I = torch.stack(I_list)
    spectrum = torch.stack(spectrum_list)
    S = torch.stack(S_list)

    return I, spectrum, S


decoder = Decoder().to(device)
decoder.load_state_dict(torch.load('checkpoints/decoder_epoch_1000.pth'))

mse = nn.MSELoss()
cosine = nn.CosineSimilarity(dim=1)

optimizer = torch.optim.Adam(decoder.parameters(), lr=0.0001)


for epoch in range(num_epoch):
    for i in range(1000):

        data, spectrum, S = generate_data(batch_size, np.random.randint(10, 20))
        data = data.to(device)
        spectrum = spectrum.to(device)
        S = S.to(device)

        pred_spectrum, pred_S = decoder(data)

        loss1 = mse(pred_spectrum, spectrum)
        loss2 = 1 - cosine(pred_spectrum, spectrum)
        loss3 = mse(pred_S, S[:, 1:].float())

        loss = loss1 + 0.75 * loss2 + 0.25 * loss3
        loss = loss.mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epoch}], Step [{i+1}/1000], Loss: {loss.item():.4f}")
  
    if (epoch + 1) % 100 == 0:
        torch.save(decoder.state_dict(), f'checkpoints/decoder_epoch_{epoch+1}.pth')
