from net import *
import torch


data = []
for wavelength in wavelengths:
    data.append((wavelength, torch.tensor([1, 0, 0, 1])))
    data.append((wavelength, torch.tensor([1, 0, 0, -1])))

encoder = Encoder(phase).to(device)
encoder.load_state_dict(torch.load("checkpoints/encoder_epoch_1000.pth"))


for wavelength, S in data:
    wavelength = wavelength.unsqueeze(0).to(device)
    S = S.unsqueeze(0).to(device)

    Ir, Il, _ = encoder(wavelength * nm, S)

    if S[0, 3] == 1:
        torch.save(Er.cpu(), f"dataset/{wavelength.item():.1f}_r_r.pt")
        torch.save(El.cpu(), f"dataset/{wavelength.item():.1f}_r_l.pt")
    else:
        torch.save(El.cpu(), f"dataset/{wavelength.item():.1f}_l_l.pt")
        torch.save(Er.cpu(), f"dataset/{wavelength.item():.1f}_l_r.pt")
