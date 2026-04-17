import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image
import matplotlib.pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

m = 1.0
cm = 1e-2
mm = 1e-3
um = 1e-6
nm = 1e-9
W = 1

d = 600 * nm
Nx = 500
Ny = 500
extent_x = Nx * d
extent_y = Ny * d
intensity = 0.1 * W / (m**2)

start_wavelength = 450
end_wavelength = 650
step = 0.4
wavelengths = torch.arange(start_wavelength, end_wavelength + step, step)

z = 1 * mm

alpha = 2.1
start_base = 450 * nm
end_base = 650 * nm

phase = nn.ParameterList([nn.Parameter(torch.randn((Ny, Nx))) for i in range(2)]).to(device)

class MonochromaticField:
    def __init__(
        self, wavelength, extent_x, extent_y, Nx, Ny, intensity=intensity, batch_size=1
    ):
        self.extent_x = extent_x
        self.extent_y = extent_y

        self.dx = extent_x / Nx
        self.dy = extent_y / Ny

        self.x = self.dx * (torch.arange(Nx) - Nx // 2).to(device)
        self.y = self.dy * (torch.arange(Ny) - Ny // 2).to(device)
        self.xx, self.yy = torch.meshgrid(self.x, self.y, indexing="xy")

        self.Nx = Nx
        self.Ny = Ny
        self.E = np.sqrt(intensity) * torch.ones((self.Ny, self.Nx)).to(device)
        self.E = torch.full((batch_size, Ny, Nx), np.sqrt(intensity), device=device)
        self.λ = wavelength
        self.z = 0

    def modulate(self, amplitude, phase):
        self.E = amplitude * self.E * torch.exp(1j * phase)

    def point_source(self, z):
        r = torch.sqrt((self.xx) ** 2 + (self.yy) ** 2 + z**2)
        self.E = 0.001 / r * torch.exp(1j * 2 * torch.pi * r / self.λ)

    def propagate(self, z):
        fft_c = torch.fft.fft2(self.E)
        c = torch.fft.fftshift(fft_c)

        fx = torch.fft.fftshift(torch.fft.fftfreq(self.Nx, d=self.dx)).to(device)
        fy = torch.fft.fftshift(torch.fft.fftfreq(self.Ny, d=self.dy)).to(device)
        fxx, fyy = torch.meshgrid(fx, fy, indexing="xy")

        argument = (2 * torch.pi) ** 2 * ((1.0 / self.λ) ** 2 - fxx**2 - fyy**2)

        tmp = torch.sqrt(torch.abs(argument))
        kz = torch.where(argument >= 0, tmp, 1j * tmp).to(device)

        self.z += z
        self.E = torch.fft.ifft2(torch.fft.ifftshift(c * torch.exp(1j * kz * z)))

    def get_intensity(self):
        return torch.real(self.E * torch.conj(self.E))


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, padding, bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.leaky = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        return self.leaky(self.bn(self.conv(x)))


class ResBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResBlock, self).__init__()
        self.conv1 = ConvBlock(
            in_channels, in_channels // 2, kernel_size=1, stride=1, padding=0
        )
        self.conv2 = ConvBlock(
            in_channels // 2, in_channels, kernel_size=3, stride=1, padding=1
        )

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        return out + residual


class Encoder(nn.Module):
    def __init__(self, parameter):
        super(Encoder, self).__init__()
        self.parameter = parameter

    def _compute_field(self, wavelength, amp1, phase1, amp2, phase2):
        F = MonochromaticField(
            wavelength, extent_x, extent_y, Nx, Ny, batch_size=len(wavelength)
        )
        F.point_source(z)
 
        F.modulate(amp1, phase1)
        F.modulate(amp2, phase2)
        F.propagate(z)
        return F.E, F.get_intensity()

    def forward(self, wavelength, S):
        wavelength = wavelength.unsqueeze(1).unsqueeze(2)
        S = S.unsqueeze(2).unsqueeze(3)

        delta = -torch.arctan2(S[:, 2], S[:, 1])
        shift = alpha * torch.pi * (wavelength - start_base) / (end_base - start_base)

        sigmoid_param = torch.sigmoid(self.parameter[0])
        amp_r = torch.sqrt((S[:, 0] + S[:, 3]) / 2)
        amp_l = torch.sqrt((S[:, 0] - S[:, 3]) / 2)

        Er1, Ir1 = self._compute_field(
            wavelength,
            amp_r,
            torch.tensor(0.0),
            sigmoid_param,
            self.parameter[1] + shift + self.parameter[2],
        )
        El1, Il1 = self._compute_field(
            wavelength,
            amp_r,
            torch.tensor(0.0),
            1 - sigmoid_param,
            self.parameter[1] + shift,
        )
        El2, Il2 = self._compute_field(
            wavelength,
            amp_l,
            delta,
            sigmoid_param,
            self.parameter[1] + shift - self.parameter[2],
        )
        Er2, Ir2 = self._compute_field(
            wavelength, amp_l, delta, 1 - sigmoid_param, self.parameter[1] + shift
        )

        Ir = Ir1 + Ir2
        Il = Il1 + Il2
        I = Ir + Il
        Er = Er1 + Er2
        El = El1 + El2

        return Er, El, I


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        self.conv1 = ConvBlock(1, 32, kernel_size=3, stride=1, padding=1)

        self.layer1 = self._make_layer(32, 64, 1)
        self.layer2 = self._make_layer(64, 128, 2)
        self.layer3 = self._make_layer(128, 256, 8)
        self.layer4 = self._make_layer(256, 512, 8)
        self.layer5 = self._make_layer(512, 1024, 4)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(1024, len(wavelengths) + 3)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def _make_layer(self, in_channels, out_channels, num_blocks):
        layers = [ConvBlock(in_channels, out_channels, 3, stride=2, padding=1)]
        for _ in range(num_blocks):
            layers.append(ResBlock(out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):

        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        spectrum = self.sigmoid(x[:, : len(wavelengths)])
        S = self.tanh(x[:, len(wavelengths) :])

        return spectrum, S
