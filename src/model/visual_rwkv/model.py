########################################################################################################
# VisualRWKV-v6 model, adapted for inference + differentiable image attacks.
#
# Upstream: https://github.com/howard-hou/VisualRWKV (VisualRWKV-v6/v6.0/src/model.py)
# Changes from upstream:
#   * dropped the PyTorch-Lightning / DeepSpeed machinery and all training code
#     (LightningModule -> nn.Module, no configure_optimizers / training_step);
#   * the WKV6 CUDA kernel is loaded lazily from this package's cuda/ directory
#     (absolute paths), so it no longer depends on the process working directory;
#   * `bidirectional_forward` is rewritten out-of-place: upstream flips slices of the
#     activation tensor in place, which corrupts autograd. We need gradients to flow
#     back to the input image for the attack, so the image span is flipped with a
#     fresh tensor (torch.cat) instead;
#   * no torch.jit (MyModule == nn.Module) to keep the graph plain and differentiable;
#   * gradient checkpointing is dropped (grad_cp is ignored) -- the attack needs a
#     full graph and we never train weights.
#
# The custom WKV6 kernel is bfloat16-only and CUDA-only, so the whole RWKV stack runs
# in bfloat16 on a GPU. Casts between the float32 image space and bfloat16 are
# differentiable, so gradients still reach the (float32) adversarial image.
########################################################################################################

import os
import math
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import CLIPVisionModel

# Model constants (match upstream src/dataset.py)
IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = -200
DEFAULT_IMAGE_TOKEN = "<image>"
STOP_TOKEN_INDEX = 261        # "\n\n"
DEFAULT_STOP_TOKEN = "\n\n"

# Upper bound on sequence length the WKV6 backward kernel can handle. kernel_backward_222
# sizes a per-thread local array as sbbbb[_T_-2], so the compiled kernel is only valid
# for runtime sequence length T <= KERNEL_CTXLEN. Prompt + image tokens + candidate stay
# well below this; RUN_CUDA_RWKV6 asserts it.
KERNEL_CTXLEN = 2048

_CUDA_DIR = Path(__file__).resolve().parent / "cuda"

# globals set by load_wkv6_kernel()
HEAD_SIZE = None
wkv6_cuda = None


def load_wkv6_kernel(head_size: int, kernel_ctxlen: int = KERNEL_CTXLEN):
    """Compile/load the WKV6 CUDA extension once. Safe to call repeatedly."""
    global HEAD_SIZE, wkv6_cuda
    if wkv6_cuda is not None:
        if HEAD_SIZE != head_size:
            raise RuntimeError(
                f"WKV6 kernel already loaded with HEAD_SIZE={HEAD_SIZE}, "
                f"cannot reload with {head_size}."
            )
        return wkv6_cuda

    from torch.utils.cpp_extension import load

    HEAD_SIZE = head_size
    wkv6_cuda = load(
        name="wkv6",
        sources=[str(_CUDA_DIR / "wkv6_op.cpp"), str(_CUDA_DIR / "wkv6_cuda.cu")],
        verbose=True,
        extra_cuda_cflags=[
            "-res-usage", "--use_fast_math", "-O3", "-Xptxas -O3",
            "--extra-device-vectorization",
            f"-D_N_={head_size}", f"-D_T_={kernel_ctxlen}",
        ],
    )
    return wkv6_cuda


class WKV_6(torch.autograd.Function):
    @staticmethod
    def forward(ctx, B, T, C, H, r, k, v, w, u):
        with torch.no_grad():
            assert r.dtype == torch.bfloat16
            assert k.dtype == torch.bfloat16
            assert v.dtype == torch.bfloat16
            assert w.dtype == torch.bfloat16
            assert u.dtype == torch.bfloat16
            assert HEAD_SIZE == C // H
            ctx.B = B
            ctx.T = T
            ctx.C = C
            ctx.H = H
            assert r.is_contiguous()
            assert k.is_contiguous()
            assert v.is_contiguous()
            assert w.is_contiguous()
            assert u.is_contiguous()
            ew = (-torch.exp(w.float())).contiguous()
            ctx.save_for_backward(r, k, v, ew, u)
            y = torch.empty((B, T, C), device=r.device, dtype=torch.bfloat16,
                            memory_format=torch.contiguous_format)
            wkv6_cuda.forward(B, T, C, H, r, k, v, ew, u, y)
            return y

    @staticmethod
    def backward(ctx, gy):
        with torch.no_grad():
            assert gy.dtype == torch.bfloat16
            B = ctx.B
            T = ctx.T
            C = ctx.C
            H = ctx.H
            assert gy.is_contiguous()
            r, k, v, ew, u = ctx.saved_tensors
            gr = torch.empty((B, T, C), device=gy.device, requires_grad=False, dtype=torch.bfloat16, memory_format=torch.contiguous_format)
            gk = torch.empty((B, T, C), device=gy.device, requires_grad=False, dtype=torch.bfloat16, memory_format=torch.contiguous_format)
            gv = torch.empty((B, T, C), device=gy.device, requires_grad=False, dtype=torch.bfloat16, memory_format=torch.contiguous_format)
            gw = torch.empty((B, T, C), device=gy.device, requires_grad=False, dtype=torch.bfloat16, memory_format=torch.contiguous_format)
            gu = torch.empty((B, C), device=gy.device, requires_grad=False, dtype=torch.bfloat16, memory_format=torch.contiguous_format)
            wkv6_cuda.backward(B, T, C, H, r, k, v, ew, u, gy, gr, gk, gv, gw, gu)
            gu = torch.sum(gu, 0).view(H, C // H)
            return (None, None, None, None, gr, gk, gv, gw, gu)


def RUN_CUDA_RWKV6(B, T, C, H, r, k, v, w, u):
    assert T <= KERNEL_CTXLEN, (
        f"sequence length {T} exceeds the compiled WKV6 kernel bound "
        f"KERNEL_CTXLEN={KERNEL_CTXLEN}; increase it and recompile."
    )
    return WKV_6.apply(B, T, C, H, r, k, v, w, u)


########################################################################################################

class RWKV_Tmix_x060(nn.Module):
    def __init__(self, args, layer_id):
        super().__init__()
        self.args = args
        self.layer_id = layer_id

        self.head_size = args.head_size_a
        self.n_head = args.dim_att // self.head_size
        assert args.dim_att % self.n_head == 0

        with torch.no_grad():
            ratio_0_to_1 = layer_id / (args.n_layer - 1)  # 0 to 1
            ratio_1_to_almost0 = 1.0 - (layer_id / args.n_layer)  # 1 to ~0
            ddd = torch.ones(1, 1, args.n_embd)
            for i in range(args.n_embd):
                ddd[0, 0, i] = i / args.n_embd

            self.time_maa_x = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))
            self.time_maa_w = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))
            self.time_maa_k = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))
            self.time_maa_v = nn.Parameter(1.0 - (torch.pow(ddd, ratio_1_to_almost0) + 0.3 * ratio_0_to_1))
            self.time_maa_r = nn.Parameter(1.0 - torch.pow(ddd, 0.5 * ratio_1_to_almost0))
            self.time_maa_g = nn.Parameter(1.0 - torch.pow(ddd, 0.5 * ratio_1_to_almost0))

            D_MIX_LORA = 32
            if args.n_embd >= 4096:
                D_MIX_LORA = 64
            self.time_maa_w1 = nn.Parameter(torch.zeros(args.n_embd, D_MIX_LORA * 5))
            self.time_maa_w2 = nn.Parameter(torch.zeros(5, D_MIX_LORA, args.n_embd).uniform_(-0.01, 0.01))

            decay_speed = torch.ones(args.dim_att)
            for n in range(args.dim_att):
                decay_speed[n] = -6 + 5 * (n / (args.dim_att - 1)) ** (0.7 + 1.3 * ratio_0_to_1)
            self.time_decay = nn.Parameter(decay_speed.reshape(1, 1, args.dim_att))

            D_DECAY_LORA = 64
            if args.n_embd >= 4096:
                D_DECAY_LORA = 128
            self.time_decay_w1 = nn.Parameter(torch.zeros(args.n_embd, D_DECAY_LORA))
            self.time_decay_w2 = nn.Parameter(torch.zeros(D_DECAY_LORA, args.dim_att).uniform_(-0.01, 0.01))

            tmp = torch.zeros(args.dim_att)
            for n in range(args.dim_att):
                zigzag = ((n + 1) % 3 - 1) * 0.1
                tmp[n] = ratio_0_to_1 * (1 - (n / (args.dim_att - 1))) + zigzag

            self.time_faaaa = nn.Parameter(tmp.reshape(self.n_head, self.head_size))

        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
        self.receptance = nn.Linear(args.n_embd, args.dim_att, bias=False)
        self.key = nn.Linear(args.n_embd, args.dim_att, bias=False)
        self.value = nn.Linear(args.n_embd, args.dim_att, bias=False)
        self.output = nn.Linear(args.dim_att, args.n_embd, bias=False)
        self.gate = nn.Linear(args.n_embd, args.dim_att, bias=False)
        self.ln_x = nn.GroupNorm(self.n_head, args.dim_att, eps=(1e-5) * (args.head_size_divisor ** 2))

    def jit_func(self, x):
        B, T, C = x.size()

        xx = self.time_shift(x) - x

        xxx = x + xx * self.time_maa_x
        xxx = torch.tanh(xxx @ self.time_maa_w1).view(B * T, 5, -1).transpose(0, 1)
        xxx = torch.bmm(xxx, self.time_maa_w2).view(5, B, T, -1)
        mw, mk, mv, mr, mg = xxx.unbind(dim=0)

        xw = x + xx * (self.time_maa_w + mw)
        xk = x + xx * (self.time_maa_k + mk)
        xv = x + xx * (self.time_maa_v + mv)
        xr = x + xx * (self.time_maa_r + mr)
        xg = x + xx * (self.time_maa_g + mg)

        r = self.receptance(xr)
        k = self.key(xk)
        v = self.value(xv)
        g = F.silu(self.gate(xg))

        ww = torch.tanh(xw @ self.time_decay_w1) @ self.time_decay_w2
        w = self.time_decay + ww

        return r, k, v, g, w

    def jit_func_2(self, x, g):
        B, T, C = x.size()
        x = x.view(B * T, C)
        x = self.ln_x(x).view(B, T, C)
        x = self.output(x * g)
        return x

    def forward(self, x):
        B, T, C = x.size()
        H = self.n_head

        r, k, v, g, w = self.jit_func(x)
        x = RUN_CUDA_RWKV6(B, T, C, H, r, k, v, w, u=self.time_faaaa)

        return self.jit_func_2(x, g)


########################################################################################################

class RWKV_CMix_x060(nn.Module):
    def __init__(self, args, layer_id):
        super().__init__()
        self.args = args
        self.layer_id = layer_id
        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))

        with torch.no_grad():
            ratio_1_to_almost0 = 1.0 - (layer_id / args.n_layer)  # 1 to ~0
            ddd = torch.ones(1, 1, args.n_embd)
            for i in range(args.n_embd):
                ddd[0, 0, i] = i / args.n_embd
            self.time_maa_k = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))
            self.time_maa_r = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))

        self.key = nn.Linear(args.n_embd, args.dim_ffn, bias=False)
        self.receptance = nn.Linear(args.n_embd, args.n_embd, bias=False)
        self.value = nn.Linear(args.dim_ffn, args.n_embd, bias=False)

    def forward(self, x):
        xx = self.time_shift(x) - x
        xk = x + xx * self.time_maa_k
        xr = x + xx * self.time_maa_r

        k = self.key(xk)
        k = torch.relu(k) ** 2
        kv = self.value(k)
        return torch.sigmoid(self.receptance(xr)) * kv


########################################################################################################

class Block(nn.Module):
    def __init__(self, args, layer_id):
        super().__init__()
        self.args = args
        self.layer_id = layer_id

        self.ln1 = nn.LayerNorm(args.n_embd)
        self.ln2 = nn.LayerNorm(args.n_embd)

        if self.layer_id == 0:
            self.ln0 = nn.LayerNorm(args.n_embd)

        self.att = RWKV_Tmix_x060(args, layer_id)
        self.ffn = RWKV_CMix_x060(args, layer_id)

    def forward(self, x):
        if self.layer_id == 0:
            x = self.ln0(x)

        x = x + self.att(self.ln1(x))
        x = x + self.ffn(self.ln2(x))

        return x


class RWKV(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.emb = nn.Embedding(args.vocab_size, args.n_embd)
        self.blocks = nn.ModuleList([Block(args, i) for i in range(args.n_layer)])
        self.ln_out = nn.LayerNorm(args.n_embd)
        self.head = nn.Linear(args.n_embd, args.vocab_size, bias=False)


class VisualRWKV(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        load_wkv6_kernel(args.head_size_a)
        self.rwkv = RWKV(args)
        self.vit = CLIPVisionModel.from_pretrained(args.vision_tower_name)
        self.vit.requires_grad_(False)
        self.proj = nn.Linear(self.vit.config.hidden_size, args.n_embd, bias=False)

    # ----- vision tower -----

    def encode_images(self, images):
        '''images: [B, N, C, H, W] (CLIP-normalized pixel values). Returns [B, n_img_tokens, n_embd].'''
        B, N, C, H, W = images.shape
        images = images.view(B * N, C, H, W)
        image_features = self.vit(images).last_hidden_state
        L, D = image_features.shape[1], image_features.shape[2]
        image_features = image_features.view(B, N, L, D)[:, 0, :, :]   # take the first image
        image_features = self.grid_pooling(image_features)
        return self.proj(image_features)

    def grid_pooling(self, image_features):
        cls_features = image_features[:, 0:1, :]
        image_features = image_features[:, 1:, :]  # drop cls token
        if self.args.grid_size == -1:  # no grid pooling
            return torch.cat((image_features, cls_features), dim=1)
        if self.args.grid_size == 0:  # take cls token
            return cls_features
        if self.args.grid_size == 1:  # global avg pooling
            return torch.cat((image_features.mean(dim=1, keepdim=True), cls_features), dim=1)
        B, L, D = image_features.shape
        H_or_W = int(L ** 0.5)
        image_features = image_features.view(B, H_or_W, H_or_W, D)
        grid_stride = H_or_W // self.args.grid_size
        image_features = F.avg_pool2d(
            image_features.permute(0, 3, 1, 2),
            padding=0, kernel_size=grid_stride, stride=grid_stride,
        )
        image_features = image_features.permute(0, 2, 3, 1).view(B, -1, D)
        return torch.cat((image_features, cls_features), dim=1)

    # ----- language model over precomputed embeddings -----

    def embed_tokens(self, ids):
        return self.rwkv.emb(ids)

    def forward_embeds(self, x, img_start=None, img_end=None):
        '''
        Run the bidirectional RWKV stack over input embeddings x: [B, T, n_embd].
        On odd layers the image span [img_start:img_end) is processed in reverse
        (the "bidirectional" image scan). Rewritten out-of-place from upstream so
        gradients flow back to x (and hence to the input image).
        Returns logits [B, T, vocab_size].
        '''
        do_bidir = img_start is not None and img_end is not None and img_end > img_start

        for i, block in enumerate(self.rwkv.blocks):
            if do_bidir and (i % 2 == 1):
                # flip the image span out-of-place, both before and after the block
                flipped = x[:, img_start:img_end, :].flip(1)
                x = torch.cat([x[:, :img_start, :], flipped, x[:, img_end:, :]], dim=1)
                x = block(x)
                flipped = x[:, img_start:img_end, :].flip(1)
                x = torch.cat([x[:, :img_start, :], flipped, x[:, img_end:, :]], dim=1)
            else:
                x = block(x)

        x = self.rwkv.ln_out(x)
        x = self.rwkv.head(x)
        return x


def build_args(config: dict) -> SimpleNamespace:
    '''Build the model-args namespace, mirroring upstream's argparse defaults.'''
    args = SimpleNamespace(
        load_model="",
        vocab_size=config.get("vocab_size", 65536),
        ctx_len=config.get("ctx_len", 2048),
        n_layer=config["n_layer"],
        n_embd=config["n_embd"],
        dim_att=config.get("dim_att", 0),
        dim_ffn=config.get("dim_ffn", 0),
        head_size_a=config.get("head_size_a", 64),
        head_size_divisor=config.get("head_size_divisor", 8),
        dropout=0.0,
        vision_tower_name=config.get("vision_tower_name", "openai/clip-vit-large-patch14-336"),
        grid_size=config.get("grid_size", -1),
        grad_cp=0,
    )
    if args.dim_att <= 0:
        args.dim_att = args.n_embd
    if args.dim_ffn <= 0:
        args.dim_ffn = int((args.n_embd * 3.5) // 32 * 32)  # default = 3.5x emb size
    return args
