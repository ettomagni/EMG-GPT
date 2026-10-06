# Adapted for EMG-GPT; provenance and license texts are listed in THIRD_PARTY_NOTICES.md.
import math
from functools import partial

import torch
from einops import rearrange
from torch import nn
from torch.nn import functional as F

from .NeuroRVQ_modules import Block, trunc_normal_
from .RVQ import ResidualVectorQuantization


class PatchEmbed(nn.Module):
    """
    Project each codebook to the patch latent space
    :param in_chans: number of input channels
    :param embed_dim: dimension of embedding space
    """

    def __init__(self, in_chans=1, embed_dim=200):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, x):
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class MultiDimentionalTemporalConv(nn.Module):
    """
    EMG to Patch Embedding - Multidimentional Temporal Filtering
    """

    def __init__(self, in_chans=1, out_chans=8):
        super().__init__()
        # --- Group 1: First-level temporal convolutions --- #
        # Branch 1: >20 Hz assuming fs=1000Hz
        self.conv1_1 = nn.Conv2d(in_chans, out_chans, kernel_size=(1, 51), padding=(0, 25))
        self.norm1_1 = nn.GroupNorm(4, out_chans)
        self.pool1_1 = nn.AvgPool2d(kernel_size=(1, 2))

        # Branch 2: >60 Hz assuming fs=1000Hz
        self.conv1_2 = nn.Conv2d(in_chans, out_chans, kernel_size=(1, 17), padding=(0, 8))
        self.norm1_2 = nn.GroupNorm(4, out_chans)
        self.pool1_2 = nn.AvgPool2d(kernel_size=(1, 2))

        # Branch 3: >125 Hz assuming fs=1000Hz
        self.conv1_3 = nn.Conv2d(in_chans, out_chans, kernel_size=(1, 8), padding=(0, 4))
        self.norm1_3 = nn.GroupNorm(4, out_chans)
        self.pool1_3 = nn.AvgPool2d(kernel_size=(1, 2))

        # Branch 4: >250 Hz assuming fs=1000Hz
        self.conv1_4 = nn.Conv2d(in_chans, out_chans, kernel_size=(1, 5), padding=(0, 2))
        self.norm1_4 = nn.GroupNorm(4, out_chans)
        self.pool1_4 = nn.AvgPool2d(kernel_size=(1, 2))
        self.gelu1 = nn.GELU()

        # --- Group 2: Second-level convolutions --- #
        self.conv2_1 = nn.Conv2d(out_chans, out_chans, kernel_size=(1, 25), padding=(0, 12))
        self.norm2_1 = nn.GroupNorm(4, out_chans)
        self.pool2_1 = nn.AvgPool2d(kernel_size=(1, 4))

        self.conv2_2 = nn.Conv2d(out_chans, out_chans, kernel_size=(1, 9), padding=(0, 4))
        self.norm2_2 = nn.GroupNorm(4, out_chans)
        self.pool2_2 = nn.AvgPool2d(kernel_size=(1, 4))

        self.conv2_3 = nn.Conv2d(out_chans, out_chans, kernel_size=(1, 4), padding=(0, 2))
        self.norm2_3 = nn.GroupNorm(4, out_chans)
        self.pool2_3 = nn.AvgPool2d(kernel_size=(1, 4))

        self.conv2_4 = nn.Conv2d(out_chans, out_chans, kernel_size=(1, 3), padding=(0, 1))
        self.norm2_4 = nn.GroupNorm(4, out_chans)
        self.pool2_4 = nn.AvgPool2d(kernel_size=(1, 4))

        self.gelu2 = nn.GELU()

    def forward(self, x):
        x = rearrange(x, "B N A T -> B (N A) T")
        B, NA, T = x.shape
        x = x.unsqueeze(1)

        # --- Group 1 --- #
        x1 = self.pool1_1(self.gelu1(self.norm1_1(self.conv1_1(x))))
        x2 = self.pool1_2(self.gelu1(self.norm1_2(self.conv1_2(x))))
        x3 = self.pool1_3(self.gelu1(self.norm1_3(self.conv1_3(x))))
        x4 = self.pool1_4(self.gelu1(self.norm1_4(self.conv1_4(x))))

        # --- Group 2 --- #
        x1 = self.pool2_1(self.gelu2(self.norm2_1(self.conv2_1(x1))))
        x2 = self.pool2_2(self.gelu2(self.norm2_2(self.conv2_2(x2))))
        x3 = self.pool2_3(self.gelu2(self.norm2_3(self.conv2_3(x3))))
        x4 = self.pool2_4(self.gelu2(self.norm2_4(self.conv2_4(x4))))

        # --- Re-arrange --- #
        x1 = rearrange(x1, "B C NA T -> B NA (T C)")
        x2 = rearrange(x2, "B C NA T -> B NA (T C)")
        x3 = rearrange(x3, "B C NA T -> B NA (T C)")
        x4 = rearrange(x4, "B C NA T -> B NA (T C)")
        return x1, x2, x3, x4


class NeuroRVQFM(nn.Module):
    """
    NeuroRVQ Foundation Model Class
    """

    def __init__(
        self,
        n_patches=256,
        patch_size=200,
        in_chans=1,
        out_chans=8,
        num_classes=5,
        embed_dim=200,
        depth=12,
        num_heads=10,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_norm=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        init_values=None,
        init_scale=0.001,
        n_global_electrodes=127,
        vocab_size=8192,
        use_as_encoder=True,
    ):

        super().__init__()

        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.patch_size = patch_size
        self.use_as_encoder = use_as_encoder
        # Not necessary - legacy code
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # To identify whether patch_embed layer is used as tokenizer/encoder or as a decoder
        if use_as_encoder:
            self.patch_embed = MultiDimentionalTemporalConv(out_chans=out_chans)
        else:
            self.patch_embed_1 = PatchEmbed(in_chans=in_chans, embed_dim=embed_dim)
            self.patch_embed_2 = PatchEmbed(in_chans=in_chans, embed_dim=embed_dim)
            self.patch_embed_3 = PatchEmbed(in_chans=in_chans, embed_dim=embed_dim)
            self.patch_embed_4 = PatchEmbed(in_chans=in_chans, embed_dim=embed_dim)

        self.pos_embed = nn.Parameter(
            torch.zeros(n_global_electrodes + 1, embed_dim), requires_grad=True
        )
        self.time_embed = nn.Parameter(torch.zeros(n_patches, embed_dim), requires_grad=True)
        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, depth)
        ]  # stochastic depth decay rule

        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_norm=qk_norm,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=nn.LayerNorm,
                    init_values=init_values,
                    window_size=None,
                )
                for i in range(depth)
            ]
        )

        self.norm = nn.Identity()
        self.fc_norm_1 = nn.LayerNorm(embed_dim)
        self.head_1 = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        self.fc_norm_2 = nn.LayerNorm(embed_dim)
        self.head_2 = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        self.fc_norm_3 = nn.LayerNorm(embed_dim)
        self.head_3 = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        self.fc_norm_4 = nn.LayerNorm(embed_dim)
        self.head_4 = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        # Initialize the weights of the network
        trunc_normal_(self.pos_embed, std=0.02)
        trunc_normal_(self.time_embed, std=0.02)
        trunc_normal_(self.cls_token, std=0.02)

        if isinstance(self.head_1, nn.Linear):
            trunc_normal_(self.head_1.weight, std=0.02)
        if isinstance(self.head_1, nn.Linear):
            self.head_1.weight.data.mul_(init_scale)
            self.head_1.bias.data.mul_(init_scale)
        if isinstance(self.head_2, nn.Linear):
            trunc_normal_(self.head_2.weight, std=0.02)
        if isinstance(self.head_2, nn.Linear):
            self.head_2.weight.data.mul_(init_scale)
            self.head_2.bias.data.mul_(init_scale)
        if isinstance(self.head_3, nn.Linear):
            trunc_normal_(self.head_3.weight, std=0.02)
        if isinstance(self.head_3, nn.Linear):
            self.head_3.weight.data.mul_(init_scale)
            self.head_3.bias.data.mul_(init_scale)
        if isinstance(self.head_4, nn.Linear):
            trunc_normal_(self.head_4.weight, std=0.02)
        if isinstance(self.head_4, nn.Linear):
            self.head_4.weight.data.mul_(init_scale)
            self.head_4.bias.data.mul_(init_scale)

        self.apply(self._init_weights)
        self.fix_init_weight()

    # Function to initialize the weights of the network
    def fix_init_weight(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    # Function to initialize the weights of the network
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x, temporal_embedding_ix, spatial_embedding_ix, return_patch_tokens=True):
        """Encode the four frequency branches with the original embedding arithmetic."""
        if not self.use_as_encoder or not return_patch_tokens:
            raise ValueError("Only tokenizer patch encoding is supported")
        branches = list(self.patch_embed(x))
        cls = self.cls_token.expand(x.shape[0], -1, -1)
        spatial_embedding_ix = F.pad(spatial_embedding_ix, (1, 0), value=0)
        spatial = self.pos_embed[spatial_embedding_ix.reshape(-1), :]
        spatial = spatial.reshape(*spatial_embedding_ix.shape, spatial.shape[-1])
        temporal = self.time_embed[temporal_embedding_ix.reshape(-1), :]
        temporal = temporal.reshape(*temporal_embedding_ix.shape, temporal.shape[-1])
        for index, branch in enumerate(branches):
            branch = torch.cat((cls, branch), dim=1) + spatial
            branch[:, 1:, :] += temporal
            branch = self.pos_drop(branch)
            for block in self.blocks:
                branch = block(branch)
            branch = self.norm(branch)[:, 1:, :]
            branches[index] = getattr(self, f"head_{index + 1}")(
                getattr(self, f"fc_norm_{index + 1}")(branch)
            )
        return *branches, None


class NeuroRVQTokenizer(nn.Module):
    """
    NeuroRVQ Tokenizer
    """

    def __init__(self, encoder_config, decoder_config, n_code, code_dim, decoder_out_dim):

        super().__init__()
        self.patch_size = encoder_config["patch_size"]
        self.code_dim = code_dim

        # Encoder layer of NeuroRVQFM
        self.encoder = NeuroRVQFM(
            n_patches=encoder_config["n_patches"],
            patch_size=encoder_config["patch_size"],
            in_chans=encoder_config["in_chans"],
            out_chans=encoder_config["out_chans_encoder"],
            num_classes=encoder_config["num_classes"],
            embed_dim=encoder_config["embed_dim"],
            depth=encoder_config["depth"],
            num_heads=encoder_config["num_heads"],
            mlp_ratio=encoder_config["mlp_ratio"],
            qkv_bias=encoder_config["qkv_bias"],
            qk_norm=partial(nn.LayerNorm, eps=1e-6),
            drop_rate=encoder_config["drop_rate"],
            attn_drop_rate=encoder_config["attn_drop_rate"],
            drop_path_rate=encoder_config["drop_path_rate"],
            init_values=encoder_config["init_values"],
            init_scale=encoder_config["init_scale"],
            n_global_electrodes=encoder_config["n_global_electrodes"],
            vocab_size=n_code,
            use_as_encoder=True,
        )

        # Decoder layer of NeuroRVQFM
        self.decoder = NeuroRVQFM(
            n_patches=decoder_config["n_patches"],
            patch_size=decoder_config["patch_size"],
            in_chans=decoder_config["in_chans"],
            out_chans=0,
            num_classes=decoder_config["num_classes"],
            embed_dim=decoder_config["embed_dim"],
            depth=decoder_config["depth"],
            num_heads=decoder_config["num_heads"],
            mlp_ratio=decoder_config["mlp_ratio"],
            qkv_bias=decoder_config["qkv_bias"],
            qk_norm=partial(nn.LayerNorm, eps=1e-6),
            drop_rate=decoder_config["drop_rate"],
            attn_drop_rate=decoder_config["attn_drop_rate"],
            drop_path_rate=decoder_config["drop_path_rate"],
            init_values=decoder_config["init_values"],
            init_scale=decoder_config["init_scale"],
            n_global_electrodes=decoder_config["n_global_electrodes"],
            vocab_size=n_code,
            use_as_encoder=False,
        )

        self.quantize_1 = ResidualVectorQuantization(
            num_quantizers=16,
            n_embed=n_code,
            embedding_dim=code_dim,
            beta=1.0,
            kmeans_init=True,
            decay=0.99,
        )
        self.quantize_2 = ResidualVectorQuantization(
            num_quantizers=16,
            n_embed=n_code,
            embedding_dim=code_dim,
            beta=1.0,
            kmeans_init=True,
            decay=0.99,
        )
        self.quantize_3 = ResidualVectorQuantization(
            num_quantizers=16,
            n_embed=n_code,
            embedding_dim=code_dim,
            beta=1.0,
            kmeans_init=True,
            decay=0.99,
        )
        self.quantize_4 = ResidualVectorQuantization(
            num_quantizers=16,
            n_embed=n_code,
            embedding_dim=code_dim,
            beta=1.0,
            kmeans_init=True,
            decay=0.99,
        )

        # Output dimension of the decoder layer
        self.decoder_out_dim = decoder_out_dim

        # Encoding head after the encoder transformer
        self.encode_task_layer_1 = nn.Sequential(
            nn.Linear(encoder_config["embed_dim"], encoder_config["embed_dim"]),
            nn.Tanh(),
            nn.Linear(encoder_config["embed_dim"], code_dim),
        )
        self.encode_task_layer_2 = nn.Sequential(
            nn.Linear(encoder_config["embed_dim"], encoder_config["embed_dim"]),
            nn.Tanh(),
            nn.Linear(encoder_config["embed_dim"], code_dim),
        )
        self.encode_task_layer_3 = nn.Sequential(
            nn.Linear(encoder_config["embed_dim"], encoder_config["embed_dim"]),
            nn.Tanh(),
            nn.Linear(encoder_config["embed_dim"], code_dim),
        )
        self.encode_task_layer_4 = nn.Sequential(
            nn.Linear(encoder_config["embed_dim"], encoder_config["embed_dim"]),
            nn.Tanh(),
            nn.Linear(encoder_config["embed_dim"], code_dim),
        )
        self.encode_task_layer_1.apply(self._init_weights)
        self.encode_task_layer_2.apply(self._init_weights)
        self.encode_task_layer_3.apply(self._init_weights)
        self.encode_task_layer_4.apply(self._init_weights)

        # Decoding heads after the decoder transformer
        self.decode_task_layer_amplitude = nn.Sequential(
            nn.Linear(4 * decoder_config["embed_dim"], decoder_config["embed_dim"]),
            nn.GELU(),
            nn.Linear(decoder_config["embed_dim"], self.decoder_out_dim),
        )
        self.decode_task_layer_angle_sin = nn.Sequential(
            nn.Linear(4 * decoder_config["embed_dim"], decoder_config["embed_dim"]),
            nn.Tanh(),
            nn.Linear(decoder_config["embed_dim"], self.decoder_out_dim),
            nn.Tanh(),
        )
        self.decode_task_layer_angle_cos = nn.Sequential(
            nn.Linear(4 * decoder_config["embed_dim"], decoder_config["embed_dim"]),
            nn.Tanh(),
            nn.Linear(decoder_config["embed_dim"], self.decoder_out_dim),
            nn.Tanh(),
        )

        # Initialize model weights
        self.decode_task_layer_amplitude.apply(self._init_weights)
        self.decode_task_layer_angle_sin.apply(self._init_weights)
        self.decode_task_layer_angle_cos.apply(self._init_weights)

    # Function to initialize the weights of the network
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def encode(self, x, temporal_embedding_ix, spatial_embedding_ix):
        batch_size, n, a, t = x.shape
        encoder_features_1, encoder_features_2, encoder_features_3, encoder_features_4, _ = (
            self.encoder(
                x,
                temporal_embedding_ix=temporal_embedding_ix,
                spatial_embedding_ix=spatial_embedding_ix,
                return_patch_tokens=True,
            )
        )

        with torch.amp.autocast("cuda", enabled=False):
            to_quantizer_features_1 = self.encode_task_layer_1(
                encoder_features_1.type_as(self.encode_task_layer_1[-1].weight)
            )
            to_quantizer_features_2 = self.encode_task_layer_2(
                encoder_features_2.type_as(self.encode_task_layer_2[-1].weight)
            )
            to_quantizer_features_3 = self.encode_task_layer_3(
                encoder_features_3.type_as(self.encode_task_layer_3[-1].weight)
            )
            to_quantizer_features_4 = self.encode_task_layer_4(
                encoder_features_4.type_as(self.encode_task_layer_4[-1].weight)
            )

        N = to_quantizer_features_1.shape[1]
        h, w = n, N // n

        # reshape tokens to feature maps for patch embed in decoder
        to_quantizer_features_1 = rearrange(
            to_quantizer_features_1, "b (h w) c -> b c h w", h=h, w=w
        ).contiguous()  # reshape for quantizer
        quantize_1, code_ind_1, loss_1, usage_ratios_1 = self.quantize_1(to_quantizer_features_1)

        to_quantizer_features_2 = rearrange(
            to_quantizer_features_2, "b (h w) c -> b c h w", h=h, w=w
        ).contiguous()  # reshape for quantizer
        quantize_2, code_ind_2, loss_2, usage_ratios_2 = self.quantize_2(to_quantizer_features_2)

        to_quantizer_features_3 = rearrange(
            to_quantizer_features_3, "b (h w) c -> b c h w", h=h, w=w
        ).contiguous()  # reshape for quantizer
        quantize_3, code_ind_3, loss_3, usage_ratios_3 = self.quantize_3(to_quantizer_features_3)

        to_quantizer_features_4 = rearrange(
            to_quantizer_features_4, "b (h w) c -> b c h w", h=h, w=w
        ).contiguous()  # reshape for quantizer
        quantize_4, code_ind_4, loss_4, usage_ratios_4 = self.quantize_4(to_quantizer_features_4)

        loss = None

        return (
            [quantize_1, quantize_2, quantize_3, quantize_4],
            [code_ind_1, code_ind_2, code_ind_3, code_ind_4],
            loss,
            [usage_ratios_1, usage_ratios_2, usage_ratios_3, usage_ratios_4],
        )
