from net import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


batch_size = 16
num_epoch = 1000

dataset = []
for wavelength in wavelengths:
    dataset.append((wavelength, torch.tensor([1, 0, 0, 1])))
    dataset.append((wavelength, torch.tensor([1, 0, 0, -1])))

dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

encoder = Encoder(phase).to(device)
decoder = Decoder().to(device)

mse = nn.MSELoss()
cosine = nn.CosineSimilarity(dim=1)

optimizer = torch.optim.Adam(
    [
        {"params": encoder.parameters(), "lr": 0.0001},
        {"params": decoder.parameters(), "lr": 0.0001},
    ]
)

for epoch in range(num_epoch):
    for i, (wavelength, S) in enumerate(dataloader):
        wavelength = wavelength.to(device)
        S = S.to(device)

        _, _, output = encoder(wavelength * nm, S)

        output = output.unsqueeze(1)

        pred_spectrum, pred_S = decoder(output)

        target = (
            F.one_hot(((wavelength - 450) / 0.4).long(), num_classes=len(wavelengths))
            .float()
            .to(device)
        )

        loss1 = mse(pred_spectrum, target)
        loss2 = 1 - cosine(pred_spectrum, target)
        loss3 = mse(pred_S, S[:, 1:].float())

        loss = loss1 + 0.75 * loss2 + 0.25 * loss3
        loss = loss.mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 100 == 0:
            print(
                f"Epoch [{epoch+1}/{num_epoch}], Step [{i+1}/{len(dataloader)}], Loss: {loss.item():.4f}"
            )

    if (epoch + 1) % 100 == 0:
        torch.save(encoder.state_dict(), f"checkpoints/encoder_epoch_{epoch+1}.pth")
        torch.save(decoder.state_dict(), f"checkpoints/decoder_epoch_{epoch+1}.pth")
