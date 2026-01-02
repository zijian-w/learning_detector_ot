from __future__ import annotations

from torch import nn


class CNNstack(nn.Module):
    def __init__(self, kernel_size = 3, hidden = 1024):
        super(CNNstack, self).__init__()

        k = self.k = kernel_size
        p = self.p = k//2
        h = self.h = hidden
        assert h%32==0
        
        c0 = 29**2
        c4 = h//4
        incr_factor = ((h//4)/c0)**0.25
        
        c1 = int(c0*incr_factor)
        c2 = int(c0*incr_factor**2)
        c3 = int(c0*incr_factor**3)
        c5 = h//8
        decr_factor = c5**0.25
        c6 = int(c5/decr_factor)
        c7 = int(c5/decr_factor**2)
        c8 = int(c5/decr_factor**3)
        
        
        act = nn.LeakyReLU()
        
        self.encoder = nn.Sequential(
            *[
                nn.Conv2d(c0, c1, kernel_size=k, padding=p),
                nn.BatchNorm2d(c1),
                nn.Conv2d(c1, c1, kernel_size=k, padding=p),
                nn.BatchNorm2d(c1),
                nn.Conv2d(c1, c1, kernel_size=k, padding=p, stride=2),
                nn.BatchNorm2d(c1),
                act,
                nn.Conv2d(c1, c2, kernel_size=k, padding=p),
                nn.BatchNorm2d(c2),
                nn.Conv2d(c2, c2, kernel_size=k, padding=p),
                nn.BatchNorm2d(c2),
                nn.Conv2d(c2, c2, kernel_size=k, padding=p, stride=2),
                nn.BatchNorm2d(c2),
                act,
                nn.Conv2d(c2, c3, kernel_size=k, padding=p),
                nn.BatchNorm2d(c3),
                nn.Conv2d(c3, c3, kernel_size=k, padding=p),
                nn.BatchNorm2d(c3),
                nn.Conv2d(c3, c3, kernel_size=k, padding=p, stride=2),
                nn.BatchNorm2d(c3),
                act,
                nn.Conv2d(c3, c4, kernel_size=k, padding=p),
                nn.BatchNorm2d(c4),
                nn.Conv2d(c4, c4, kernel_size=k, padding=p),
                nn.BatchNorm2d(c4),
                nn.Conv2d(c4, c4, kernel_size=k, padding=p, stride=2),
                nn.BatchNorm2d(c4),
                act
            ]
        )
        
        self.decoder = nn.Sequential(
            *[
                nn.ConvTranspose3d(c5,c5,kernel_size=k,padding=p,stride=(3,3,2)),
                nn.BatchNorm3d(c5),
                nn.Conv3d(c5,c5,kernel_size=k,padding=p),
                nn.BatchNorm3d(c5),
                nn.Conv3d(c5,c6,kernel_size=k,padding=p),
                nn.BatchNorm3d(c6),
                act,
                nn.ConvTranspose3d(c6,c6,kernel_size=k,padding=p,stride=2),
                nn.BatchNorm3d(c6),
                nn.Conv3d(c6,c6,kernel_size=k,padding=p),
                nn.BatchNorm3d(c6),
                nn.Conv3d(c6,c7,kernel_size=k,padding=p),
                nn.BatchNorm3d(c7),
                act,
                nn.ConvTranspose3d(c7,c7,kernel_size=(k+2,k+2,k+1),padding=p,stride=2),
                nn.BatchNorm3d(c7),
                nn.Conv3d(c7,c7,kernel_size=k,padding=p),
                nn.BatchNorm3d(c7),
                nn.Conv3d(c7,c8,kernel_size=k,padding=p),
                nn.BatchNorm3d(c8),
                act,
                nn.ConvTranspose3d(c8,c8,kernel_size=k,padding=p,stride=2),
                nn.BatchNorm3d(c8),
                nn.Conv3d(c8,c8,kernel_size=k,padding=p),
                nn.BatchNorm3d(c8),
                nn.Conv3d(c8,1,kernel_size=k,padding=p)
            ]
        )

    def forward(self, x):
        x = x.view(-1,29*29,29,29)
        x = self.encoder(x)
        x = x.view(-1,2,2,2,self.h//32,2,2)
        x = x.view(-1,2,2,2,self.h//8).permute(0,4,1,2,3).contiguous()
        x = self.decoder(x)
        return x.squeeze(1)

    def name(self):
        return f'stack-CNN-{self.k}-{self.h}'
