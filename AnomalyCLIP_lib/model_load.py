from scipy.spatial.distance import mahalanobis
import hashlib
import os
import urllib
import warnings
from typing import Union, List
from pkg_resources import packaging
import torch.nn.functional as F

import torch
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from tqdm import tqdm
import numpy as np
from scipy.ndimage import gaussian_filter

from .build_model import build_model
from .simple_tokenizer import SimpleTokenizer as _Tokenizer
from torchvision.transforms import InterpolationMode
from .AnomalyCLIP import Cmpl

if packaging.version.parse(torch.__version__) < packaging.version.parse("1.7.1"):
    warnings.warn("PyTorch version 1.7.1 or higher is recommended")


__all__ = ["available_models", "load", "_normalization",
           "_get_similarity_map",  "_compute_similarity", "cmpl_global_results", "compute_global_results", "compute_local_results", "cmpl_local_results", "get_global_scores"]
_tokenizer = _Tokenizer()

_MODELS = {
    "ViT-L/14@336px": "https://openaipublic.azureedge.net/clip/models/3035c92b350959924f9f00213499208652fc7ea050643e8b385c2dac08641f02/ViT-L-14-336px.pt",
}

from get_predefined_text import TextSlice, predefined_text_descriptions

def _download(
        url: str,
        cache_dir: Union[str, None] = None,
):

    if not cache_dir:
        cache_dir = os.path.expanduser("./cache/clip")
    os.makedirs(cache_dir, exist_ok=True)
    filename = os.path.basename(url)

    if 'openaipublic' in url:
        expected_sha256 = url.split("/")[-2]
    elif 'mlfoundations' in url:
        expected_sha256 = os.path.splitext(filename)[0].split("-")[-1]
    else:
        expected_sha256 = ''

    download_target = os.path.join(cache_dir, filename)

    if os.path.exists(download_target) and not os.path.isfile(download_target):
        raise RuntimeError(f"{download_target} exists and is not a regular file")

    if os.path.isfile(download_target):
        if expected_sha256:
            if hashlib.sha256(open(download_target, "rb").read()).hexdigest().startswith(expected_sha256):
                return download_target
            else:
                warnings.warn(f"{download_target} exists, but the SHA256 checksum does not match; re-downloading the file")
        else:
            return download_target

    with urllib.request.urlopen(url) as source, open(download_target, "wb") as output:
        with tqdm(total=int(source.headers.get("Content-Length")), ncols=80, unit='iB', unit_scale=True) as loop:
            while True:
                buffer = source.read(8192)
                if not buffer:
                    break

                output.write(buffer)
                loop.update(len(buffer))

    if expected_sha256 and not hashlib.sha256(open(download_target, "rb").read()).hexdigest().startswith(expected_sha256):
        raise RuntimeError(f"Model has been downloaded but the SHA256 checksum does not not match")

    return download_target


def _convert_image_to_rgb(image):
    return image.convert("RGB")

def _cmpl_similarity(Cmpl, items, layers=None):
    if layers is None:
        layers = list(range(12))

    features = {}

    def hook_fn(output, layer_idx):
        features[layer_idx] = output

    def make_hook(layer_idx):
        return lambda module, input, output: hook_fn(output, layer_idx)

    hooks = []
    for i, block in enumerate(Cmpl.blocks):
        if i in layers:
            hooks.append(block.register_forward_hook(make_hook(i)))

    with torch.no_grad():
        _ = Cmpl(items["cmpl"].to("cuda"))

    for hook in hooks:
        hook.remove()

    def get_feat(layer_idx):
        return features[layer_idx].squeeze(0)[1:].detach().cpu().numpy()

    def compute_anomaly(feat):
        mean = feat.mean(axis=0)
        cov = np.cov(feat, rowvar=False)
        inv_cov = np.linalg.pinv(cov)
        return np.array([mahalanobis(x, mean, inv_cov) for x in feat])

    layer_weights = [(4, 0.3), (9, 0.5), (11, 0.2)]
    anomaly_map = sum(
        w * compute_anomaly(get_feat(layer_idx)).reshape(37, 37)
        for layer_idx, w in layer_weights
    )
    anomaly_map = gaussian_filter(anomaly_map, sigma=1)
    anomaly_map = Image.fromarray(anomaly_map).resize((336, 336), resample=Image.BILINEAR)
    anomaly_map = np.array(anomaly_map)
    denom = anomaly_map.max() - anomaly_map.min()
    anomaly_map = (anomaly_map - anomaly_map.min()) / (denom + 1e-12)
    return torch.tensor(anomaly_map).unsqueeze(0)

def _compute_text_sparsity(image_features, text_features, alpha=0.0):
    spatial_feats = image_features[:, 1:, :]
    sparse_score = torch.norm(spatial_feats, p=1, dim=-1) / 768
    mean = sparse_score.mean(dim=1, keepdim=True)
    std = sparse_score.std(dim=1, keepdim=True)
    sparse_score = (sparse_score - mean) / (std + 1e-8)
    sparse_score = torch.cat([torch.zeros_like(image_features[:, :1, 0]), sparse_score], dim=1)
    text_score = image_features @ text_features.t() / 0.07
    text_score = text_score[..., 0] - text_score[..., 1]
    text_score = (text_score - text_score.min(dim=1, keepdim=True)[0]) / \
                 (text_score.max(dim=1, keepdim=True)[0] - text_score.min(dim=1, keepdim=True)[0] + 1e-8)
    scores = alpha * sparse_score + (1 - alpha) * text_score
    return torch.stack([scores, -scores], dim=-1).softmax(dim=-1)

def _detect_anomalies(features, image_size=336, patch_size=14, original_size=(336, 336)):
    features = features.squeeze(0)[1:].cuda()
    grid_size = image_size // patch_size
    mean = torch.mean(features, dim=0)
    centered = features - mean.unsqueeze(0)
    cov = torch.mm(centered.t(), centered) / (centered.size(0) - 1)
    inv_cov = torch.linalg.pinv(cov)
    diff = features - mean.unsqueeze(0)
    anomaly_scores = torch.sqrt(torch.sum(diff @ inv_cov * diff, dim=1))
    anomaly_scores = anomaly_scores.reshape(grid_size, grid_size).cpu().numpy()
    anomaly_scores = torch.from_numpy(anomaly_scores).float().cuda()
    anomaly_scores = anomaly_scores.unsqueeze(0).unsqueeze(0)
    anomaly_map = F.interpolate(anomaly_scores, size=original_size, mode='bilinear',
                                align_corners=False).squeeze().cpu().numpy()
    min_val = anomaly_map.min()
    max_val = anomaly_map.max()
    if max_val > min_val:
        anomaly_map = (anomaly_map - min_val) / (max_val - min_val)
    return torch.tensor(anomaly_map)

def _back_to_3d(d2_similarity_map, d2_3d_cor, non_zero_index):
    h = torch.sqrt(torch.tensor(d2_3d_cor.shape[2])).int()
    w = h
    b, nv, num_points, _ = d2_3d_cor.shape
    xx = d2_3d_cor[:, :, :, 0].reshape(-1).long()
    yy = d2_3d_cor[:, :, :, 1].reshape(-1).long()
    nbatch = torch.repeat_interleave(torch.arange(0, b * nv)[:, None], num_points).reshape(-1, ).cuda().long()
    d2_similarity_map = d2_similarity_map.permute(0, 3, 1, 2)
    point_logits = d2_similarity_map[nbatch, :, yy, xx]
    point_logits = point_logits.reshape(b, nv, num_points, 2)
    vweights = torch.ones((1, nv, 1, 1))
    vweights = vweights.reshape(1, -1, 1, 1).to(point_logits.device)
    is_seen = d2_3d_cor[:, :, :, 2].reshape(b, nv, num_points, 1)
    point_logits = point_logits * vweights * is_seen * non_zero_index
    mask = is_seen.bool() | (~non_zero_index.bool())
    point_logits = point_logits.sum(1) / (mask.sum(1))
    point_logits = point_logits.reshape(b, h, w, 2)
    return point_logits

def _transform(n_px):
    return Compose([
        Resize((n_px, n_px), interpolation=InterpolationMode.BICUBIC),
        #CenterCrop(n_px), # rm center crop to explain whole image
        _convert_image_to_rgb,
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])


def available_models() -> List[str]:
    """Returns the names of available CLIP models"""
    return list(_MODELS.keys())


def load_state_dict(checkpoint_path: str, map_location='cpu'):
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    if next(iter(state_dict.items()))[0].startswith('module'):
        state_dict = {k[7:]: v for k, v in state_dict.items()}
    return state_dict

def load_checkpoint(model, checkpoint_path, strict=True):
    state_dict = load_state_dict(checkpoint_path)
    # detect old format and make compatible with new format
    if 'positional_embedding' in state_dict and not hasattr(model, 'positional_embedding'):
        state_dict = convert_to_custom_text_state_dict(state_dict)
    resize_pos_embed(state_dict, model)
    incompatible_keys = model.load_state_dict(state_dict, strict=strict)
    return incompatible_keys

def load(name: str, device: Union[str, torch.device] = "cuda" if torch.cuda.is_available() else "cpu", design_details = None, jit: bool = False, download_root: str = None):
    """Load a CLIP model

    Parameters
    ----------
    name : str
        A model name listed by `clip.available_models()`, or the path to a model checkpoint containing the state_dict

    device : Union[str, torch.device]
        The device to put the loaded model

    jit : bool
        Whether to load the optimized JIT model or more hackable non-JIT model (default).

    download_root: str
        path to download the model files; by default, it uses "~/.cache/clip"

    Returns
    -------
    model : torch.nn.Module
        The CLIP model

    preprocess : Callable[[PIL.Image], torch.Tensor]
        A torchvision transform that converts a PIL image into a tensor that the returned model can take as its input
    """
    print("name", name)
    if name in _MODELS:
        model_path = _download(_MODELS[name], download_root or os.path.expanduser("./cache/clip"))
    elif os.path.isfile(name):
        model_path = name
    else:
        raise RuntimeError(f"Model {name} not found; available models = {available_models()}")

    with open(model_path, 'rb') as opened_file:
        try:
            # loading JIT archive
            model = torch.jit.load(opened_file, map_location=device if jit else "cpu").eval()
            state_dict = None
        except RuntimeError:
            # loading saved state dict
            if jit:
                warnings.warn(f"File {model_path} is not a JIT archive. Loading as a state dict instead")
                jit = False
            state_dict = torch.load(opened_file, map_location="cpu")

    if not jit:
        model = build_model(name, state_dict or model.state_dict(), design_details).to(device)
        if str(device) == "cpu":
            model.float()
        return model, _transform(model.visual.input_resolution)

    # patch the device names
    device_holder = torch.jit.trace(lambda: torch.ones([]).to(torch.device(device)), example_inputs=[])
    device_node = [n for n in device_holder.graph.findAllNodes("prim::Constant") if "Device" in repr(n)][-1]

    def patch_device(module):
        try:
            graphs = [module.graph] if hasattr(module, "graph") else []
        except RuntimeError:
            graphs = []

        if hasattr(module, "forward1"):
            graphs.append(module.forward1.graph)

        for graph in graphs:
            for node in graph.findAllNodes("prim::Constant"):
                if "value" in node.attributeNames() and str(node["value"]).startswith("cuda"):
                    node.copyAttributes(device_node)

    model.apply(patch_device)
    patch_device(model.encode_image)
    patch_device(model.encode_text)

    # patch dtype to float32 on CPU
    if str(device) == "cpu":
        float_holder = torch.jit.trace(lambda: torch.ones([]).float(), example_inputs=[])
        float_input = list(float_holder.graph.findNode("aten::to").inputs())[1]
        float_node = float_input.node()

        def patch_float(module):
            try:
                graphs = [module.graph] if hasattr(module, "graph") else []
            except RuntimeError:
                graphs = []

            if hasattr(module, "forward1"):
                graphs.append(module.forward1.graph)

            for graph in graphs:
                for node in graph.findAllNodes("aten::to"):
                    inputs = list(node.inputs())
                    for i in [1, 2]:  # dtype can be the second or third argument to aten::to()
                        if inputs[i].node()["value"] == 5:
                            inputs[i].node().copyAttributes(float_node)

        model.apply(patch_float)
        patch_float(model.encode_image)
        patch_float(model.encode_text)

        model.float()

    return model, _transform(model.input_resolution.item())


def _get_similarity_map(sm, shape):
    side = int(sm.shape[1] ** 0.5)
    sm = sm.reshape(sm.shape[0], side, side, -1).permute(0, 3, 1, 2)
    sm = torch.nn.functional.interpolate(sm, shape, mode='bilinear')
    sm = sm.permute(0, 2, 3, 1)
    return sm

def _compute_similarity(image_features, text_features):
    b, n_t, n_i, c = image_features.shape[0], text_features.shape[0], image_features.shape[1], image_features.shape[2]
    feats = image_features.reshape(b, n_i, 1, c) * text_features.reshape(1, 1, n_t, c)
    similarity = feats.sum(-1)
    return (similarity/0.07).softmax(-1)

def cmpl_global_results(Visual_Embeddings, Text_pos, Text_neg, args):
    def _compute_probs(img_feat, pos_slice, neg_slice):
        text_feat = TextSlice(pos_slice, neg_slice).get_text_feature()
        logits = img_feat.unsqueeze(1) @ text_feat.permute(0, 2, 1)
        return (logits / 0.07).softmax(-1)[:, 0, 1]

    return [
        torch.stack(
            torch.chunk(
                _compute_probs(
                    Visual_Embeddings[0],
                    Text_pos[args.p0 + args.p1:],
                    Text_neg[args.n0 + args.n1:]
                ),
                9, dim=0
            ),
            dim=1
        ).mean(1),
        _compute_probs(
            Visual_Embeddings[1],
            Text_pos[args.p0:args.p0 + args.p1],
            Text_neg[args.n0:args.n0 + args.n1]
        )
    ]

def _normalization(Features):
    return Features / Features.norm(dim=-1, keepdim=True)

def compute_global_results(Features, Text_pos, Text_neg, d2_3d_cor, non_zero_index, args):
    similarity = _compute_similarity(_normalization(Features), TextSlice(Text_pos, Text_neg).get_text_feature()[0])
    similarity = _back_to_3d(_get_similarity_map(similarity[:, 1:, :], args.image_size), d2_3d_cor, non_zero_index)[..., 1]
    return torch.stack([torch.from_numpy(gaussian_filter(i, sigma=args.test_sigma)) for i in similarity.detach().cpu()], dim=0).max()

def compute_local_results(Hybrid_Patch_Features, Text_pos, Text_neg, d2_3d_cor, non_zero_index, args):
    anomaly_map_a = [torch.as_tensor(_detect_anomalies(Hybrid_Patch_Features[i], original_size=(336, 336)), device="cuda") for i in range(9)]
    anomaly_map_a = (torch.stack(anomaly_map_a, dim=0).unsqueeze(-1).expand(-1, -1, -1, 2))
    anomaly_map_b = _compute_text_sparsity(_normalization(Hybrid_Patch_Features), TextSlice(Text_pos, Text_neg).get_text_feature()[0], alpha=0.3)
    anomaly_map_b = _get_similarity_map(anomaly_map_b[:, 1:, :], args.image_size)
    anomaly_map = args.beta * anomaly_map_b + (1 - args.beta) * anomaly_map_a
    anomaly_map = _back_to_3d(anomaly_map, d2_3d_cor, non_zero_index)[..., 1]
    return torch.stack(
        [torch.from_numpy(gaussian_filter(i, sigma=args.test_sigma)) for i in anomaly_map.detach().cpu()],
        dim=0)

def cmpl_local_results(items, Visual_Embeddings, Text_pos, Text_neg, args):
    similarity = _compute_similarity(_normalization(Visual_Embeddings[2]), TextSlice(Text_pos, Text_neg).get_text_feature()[0])
    similarity = _get_similarity_map(similarity[:, 1:, :], args.image_size)
    similarity = (similarity[..., 1] + 1 - similarity[..., 0]) / 2.0
    return [torch.stack([torch.from_numpy(gaussian_filter(i, sigma=args.test_sigma)) for i in _cmpl_similarity(Cmpl, items).detach().cpu()], dim=0), torch.stack([torch.from_numpy(gaussian_filter(i, sigma=args.test_sigma)) for i in similarity.detach().cpu()], dim=0)]

def get_global_scores(Hybrid_results, Global_results, weights):
    terms = []
    terms.append(weights[0] * Hybrid_results)
    for w, score in zip(weights[1:], Global_results):
        terms.append(w * score.detach().cpu())
    return sum(terms)

