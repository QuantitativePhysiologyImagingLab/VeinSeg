"""
Self-contained model architecture for VeinSeg inference.
No nnunetv2 dependency required.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def _center_crop_or_pad_3d(t: torch.Tensor, target_spatial):
    D, H, W = t.shape[-3:]
    Dt, Ht, Wt = target_spatial
    sd = max((D - Dt) // 2, 0); sh = max((H - Ht) // 2, 0); sw = max((W - Wt) // 2, 0)
    t = t[..., sd:sd + min(Dt, D), sh:sh + min(Ht, H), sw:sw + min(Wt, W)]
    D2, H2, W2 = t.shape[-3:]
    pd = max(Dt - D2, 0); ph = max(Ht - H2, 0); pw = max(Wt - W2, 0)
    pad = (pw // 2, pw - pw // 2, ph // 2, ph - ph // 2, pd // 2, pd - pd // 2)
    if any(p > 0 for p in pad):
        t = F.pad(t, pad, mode="constant", value=0.0)
    return t


def _match_spatial(a: torch.Tensor, b: torch.Tensor):
    Da, Ha, Wa = a.shape[-3:]; Db, Hb, Wb = b.shape[-3:]
    target = (min(Da, Db), min(Ha, Hb), min(Wa, Wb))
    return _center_crop_or_pad_3d(a, target), _center_crop_or_pad_3d(b, target)


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel, stride, padding, transposed=False, out_pad=0):
        super().__init__()
        if transposed:
            conv = nn.ConvTranspose3d(in_ch, out_ch, kernel, stride, padding, output_padding=out_pad)
        else:
            conv = nn.Conv3d(in_ch, out_ch, kernel, stride, padding)
        self.block = nn.Sequential(conv, nn.InstanceNorm3d(out_ch), nn.ELU(inplace=True))

    def forward(self, x):
        return self.block(x)


class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(nn.Conv3d(F_g, F_int, 1), nn.InstanceNorm3d(F_int), nn.ELU(inplace=True))
        self.W_x = nn.Sequential(nn.Conv3d(F_l, F_int, 1), nn.InstanceNorm3d(F_int), nn.ELU(inplace=True))
        self.psi = nn.Sequential(nn.Conv3d(F_int, 1, 1), nn.Sigmoid())

    def forward(self, g, x):
        g1, x1 = _match_spatial(self.W_g(g), self.W_x(x))
        psi = self.psi(g1 + x1)
        if psi.shape[-3:] != x.shape[-3:]:
            psi = _center_crop_or_pad_3d(psi, x.shape[-3:])
        return x * psi


class AttentionDecoder(nn.Module):
    def __init__(self, bf=64, out_ch=2, deep_supervision=False):
        super().__init__()
        self.deep_supervision = deep_supervision

        self.up4  = ConvBlock(bf * 16, bf * 8, 4, 2, 1, transposed=True)
        self.att4 = AttentionGate(bf * 8, bf * 4, bf * 4)
        self.up3  = ConvBlock(bf * 12, bf * 4, 4, 2, 1, transposed=True)
        self.att3 = AttentionGate(bf * 4, bf * 2, bf * 2)
        self.up2  = ConvBlock(bf * 6, bf * 2, 4, 2, 1, transposed=True)
        self.att2 = AttentionGate(bf * 2, bf, bf)
        self.up1  = ConvBlock(bf * 3, bf, 4, 2, 1, transposed=True)
        self.final = nn.Conv3d(bf, out_ch, 1)

        if deep_supervision:
            self.ds3 = nn.Conv3d(bf * 6, out_ch, 1)
            self.ds2 = nn.Conv3d(bf * 3, out_ch, 1)
            self.ds1 = nn.Conv3d(bf, out_ch, 1)

    def forward(self, b, x4, x3, x2, x1):
        u4 = self.up4(b)
        if u4.shape[-3:] != x3.shape[-3:]:
            u4 = _center_crop_or_pad_3d(u4, x3.shape[-3:])
        u4 = torch.cat([u4, self.att4(u4, x3)], dim=1)

        u3 = self.up3(u4)
        if u3.shape[-3:] != x2.shape[-3:]:
            u3 = _center_crop_or_pad_3d(u3, x2.shape[-3:])
        u3 = torch.cat([u3, self.att3(u3, x2)], dim=1)

        u2 = self.up2(u3)
        if u2.shape[-3:] != x1.shape[-3:]:
            u2 = _center_crop_or_pad_3d(u2, x1.shape[-3:])
        u2 = torch.cat([u2, self.att2(u2, x1)], dim=1)

        u1 = self.up1(u2)
        out = self.final(u1)

        if self.deep_supervision and self.training:
            return [out,
                    F.interpolate(self.ds1(u1), size=[s // 2 for s in out.shape[2:]], mode='trilinear', align_corners=False),
                    F.interpolate(self.ds2(u2), size=[s // 4 for s in out.shape[2:]], mode='trilinear', align_corners=False),
                    F.interpolate(self.ds3(u3), size=[s // 8 for s in out.shape[2:]], mode='trilinear', align_corners=False)]
        return out


class FiLMLayer(nn.Module):
    def __init__(self, embed_dim, num_channels):
        super().__init__()
        self.fc = nn.Linear(embed_dim, 2 * num_channels)
        nn.init.zeros_(self.fc.weight); nn.init.zeros_(self.fc.bias)

    def forward(self, x, emb):
        gb = self.fc(emb)
        gamma, beta = gb.chunk(2, dim=1)
        shape = (x.shape[0], x.shape[1]) + (1,) * (x.ndim - 2)
        return (gamma.view(shape) + 1.0) * x + beta.view(shape)


class UNetWithAttention(nn.Module):
    def __init__(self, in_channels=1, out_channels=2, patch_size=(128, 160, 112),
                 deep_supervision=False, base_features=64, num_domains=6, domain_embed_dim=32):
        super().__init__()
        self.deep_supervision = deep_supervision
        self.num_domains = num_domains
        self.default_domain_idx = 0
        self.default_field_idx  = 0

        bf = base_features
        self.enc1      = ConvBlock(in_channels, bf,      4, 2, 1)
        self.enc2      = ConvBlock(bf,          bf * 2,  4, 2, 1)
        self.enc3      = ConvBlock(bf * 2,      bf * 4,  4, 2, 1)
        self.enc4      = ConvBlock(bf * 4,      bf * 8,  4, 2, 1)
        self.bottleneck = ConvBlock(bf * 8,     bf * 16, 3, 1, 1)
        self.decoder   = AttentionDecoder(bf, out_channels, deep_supervision)

        if num_domains > 1:
            self.domain_embed = nn.Embedding(num_domains, domain_embed_dim)
            self.field_embed  = nn.Embedding(2, domain_embed_dim)
            nn.init.zeros_(self.field_embed.weight)
            self.film_enc1   = FiLMLayer(domain_embed_dim, bf)
            self.film_enc2   = FiLMLayer(domain_embed_dim, bf * 2)
            self.film_enc3   = FiLMLayer(domain_embed_dim, bf * 4)
            self.film_enc4   = FiLMLayer(domain_embed_dim, bf * 8)
            self.film_bottle = FiLMLayer(domain_embed_dim, bf * 16)
        else:
            self.domain_embed = None
            self.field_embed  = None

    def forward(self, x, domain_idx=None, field_idx=None, extra_emb=None, x1_transform=None):
        """
        extra_emb: optional (B, domain_embed_dim) tensor added to the FiLM embedding.
        x1_transform: optional callable applied to the first encoder stage's output
        (post-FiLM) before it feeds enc2 and the first skip connection.
        """
        if not self.training:
            self.decoder.deep_supervision = False
        else:
            self.decoder.deep_supervision = self.deep_supervision

        if self.domain_embed is not None:
            if domain_idx is None:
                domain_idx = torch.full((x.shape[0],), self.default_domain_idx, dtype=torch.long, device=x.device)
            if field_idx is None:
                field_idx = torch.full((x.shape[0],), self.default_field_idx, dtype=torch.long, device=x.device)
            emb = self.domain_embed(domain_idx) + self.field_embed(field_idx)
            if extra_emb is not None:
                emb = emb + extra_emb
            x1 = self.film_enc1(self.enc1(x), emb)
            if x1_transform is not None:
                x1 = x1_transform(x1)
            x2 = self.film_enc2(self.enc2(x1), emb)
            x3 = self.film_enc3(self.enc3(x2), emb)
            x4 = self.film_enc4(self.enc4(x3), emb)
            b  = self.film_bottle(self.bottleneck(x4), emb)
        else:
            x1 = self.enc1(x)
            if x1_transform is not None:
                x1 = x1_transform(x1)
            x2 = self.enc2(x1); x3 = self.enc3(x2); x4 = self.enc4(x3)
            b  = self.bottleneck(x4)

        return self.decoder(b, x4, x3, x2, x1)


class PriorGate(nn.Module):
    """gate = gamma * sigmoid(conv1x1(priors)); applied as img * (1 + gate)."""
    def __init__(self, n_priors=2):
        super().__init__()
        self.to_gate = nn.Conv3d(n_priors, 1, kernel_size=1, bias=True)
        self.gamma = nn.Parameter(torch.tensor(1.0))

    def forward(self, priors):
        return self.gamma * torch.sigmoid(self.to_gate(priors))

    def constant_gate(self, ref):
        """Gate for an all-zero prior input (used when no priors are supplied)."""
        const = self.gamma * torch.sigmoid(self.to_gate.bias)
        return const.view(1, 1, 1, 1, 1).to(device=ref.device, dtype=ref.dtype)


class R2starAdapter(nn.Module):
    """Residual conv block after enc1, applied only to R2* samples."""
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=3, padding=1),
            nn.InstanceNorm3d(channels),
            nn.ELU(inplace=True),
            nn.Conv3d(channels, channels, kernel_size=3, padding=1),
        )
        nn.init.zeros_(self.block[-1].weight); nn.init.zeros_(self.block[-1].bias)

    def forward(self, x):
        return self.block(x)


class PriorGatedSingleChannelUNet(UNetWithAttention):
    """
    Single-channel encoder: only the primary image (QSM or R2*) enters enc1,
    modulated by a gate computed from the prior channels (local field, Frangi).
    """
    def __init__(self, out_channels=2, patch_size=(128, 160, 112), deep_supervision=False,
                 num_domains=6, domain_embed_dim=32, n_priors=2, base_features=64):
        super().__init__(in_channels=1, out_channels=out_channels, patch_size=patch_size,
                         deep_supervision=deep_supervision, base_features=base_features,
                         num_domains=num_domains, domain_embed_dim=domain_embed_dim)
        self.prior_gate = PriorGate(n_priors=n_priors)
        self.r2star_adapter = R2starAdapter(base_features)
        self.pos_mlp = nn.Sequential(
            nn.Linear(3, domain_embed_dim),
            nn.SiLU(),
            nn.Linear(domain_embed_dim, domain_embed_dim),
        )
        nn.init.zeros_(self.pos_mlp[-1].weight); nn.init.zeros_(self.pos_mlp[-1].bias)

    def forward(self, img, priors=None, domain_idx=None, field_idx=None, pos=None):
        gate = self.prior_gate(priors) if priors is not None else self.prior_gate.constant_gate(img)
        gated = img * (1.0 + gate)
        extra_emb = self.pos_mlp(pos) if pos is not None else None

        if domain_idx is None:
            domain_idx = torch.full((img.shape[0],), self.default_domain_idx,
                                    dtype=torch.long, device=img.device)
        r2star_gate = (domain_idx == R2STAR_DOMAIN_IDX).to(img.dtype).view(-1, 1, 1, 1, 1)

        def _r2star_x1_transform(x1):
            return x1 + r2star_gate * self.r2star_adapter(x1)

        return super().forward(gated, domain_idx=domain_idx, field_idx=field_idx,
                               extra_emb=extra_emb, x1_transform=_r2star_x1_transform)


class PriorGatedSingleChannelUNetInfer(PriorGatedSingleChannelUNet):
    """
    nnUNetPredictor entry point: takes the 3-channel preprocessed tensor
    [primary, local field, Frangi] and splits it into image + priors.
    """
    def forward(self, x, domain_idx=None, field_idx=None, pos=None):
        y = super().forward(x[:, 0:1], priors=x[:, 1:3],
                            domain_idx=domain_idx, field_idx=field_idx, pos=pos)
        if isinstance(y, (list, tuple)):
            return y[0]
        return y


DOMAIN_METHODS = ['TGV', 'medi', 'l1', 'star', 'ilsqr', 'R2star']
METHOD_TO_IDX = {m: i for i, m in enumerate(DOMAIN_METHODS)}
R2STAR_DOMAIN_IDX = METHOD_TO_IDX['R2star']
