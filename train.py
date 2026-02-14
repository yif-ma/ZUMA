import AnomalyCLIP_lib
from dataset import Dataset
from logger import get_logger
import random
from tabulate import tabulate
from utils import get_transform
from metrics_utils import calculate_au_pro
import clip
from metrics import image_level_metrics, pixel_level_metrics
from tqdm import tqdm
from get_predefined_text import TextSlice, predefined_text_descriptions
import torch
import numpy as np
from scipy.ndimage import gaussian_filter
import argparse
import yaml
from types import SimpleNamespace

def load_yaml_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"Config must be a dict, got {type(cfg)}")
    return cfg

def _minmax_norm(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return (x - x.min()) / (x.max() - x.min() + eps)

def process_pixel_maps(Local_results, Hybrid_local_results, args):
    pc_map  = Local_results[1]
    color_anomaly_maps = Local_results[0]
    anomaly_maps = Hybrid_local_results

    if args.test_dataset == 'mvtec_pc_3d_rgb':
        pc_map = _minmax_norm(pc_map)
        color_anomaly_maps = _minmax_norm(color_anomaly_maps)
        anomaly_maps = _minmax_norm(anomaly_maps)

    integrate_anomaly_maps = (pc_map + color_anomaly_maps + anomaly_maps) / 3.0

    integrated_np = integrate_anomaly_maps.detach().cpu().numpy()
    integrated_np = np.stack(
        [gaussian_filter(im, sigma=args.test_sigma) for im in integrated_np],
        axis=0
    )
    return torch.from_numpy(integrated_np)

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def test(args):
    logger = get_logger(args.test_save_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    AnomalyCLIP_parameters = {"Prompt_length": args.test_n_ctx, "learnabel_text_embedding_depth": args.test_depth, "learnabel_text_embedding_length": args.test_t_n_ctx}
    ACLIP_Image_Ecnoder, _ = AnomalyCLIP_lib.load("ViT-L/14@336px", device=device, design_details=AnomalyCLIP_parameters)
    ACLIP_Image_Ecnoder.eval().to(device)
    ACLIP_Image_Ecnoder.visual.DAPM_replace(DPAM_layer=20)
    ACLIP_Image_Ecnoder.to(device)
    Clip_Text_Encoder, _ = clip.load("ViT-L/14@336px", device=device)
    preprocess, target_transform, target_transform_pc = get_transform(args)
    test_data = Dataset(root=args.test_data_path, dataset_name=args.test_dataset, transform=preprocess,
                        target_transform=target_transform, target_transform_pc=target_transform_pc, mode='test',
                        is_all=True, point_size=args.point_size)
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False)
    obj_list = test_data.obj_list
    results = {}

    for obj in obj_list:
        results[obj] = {}
        results[obj]['gt_sp'] = []
        results[obj]['pr_sp'] = []
        results[obj]['color_pr_sp'] = []
        results[obj]['integrate_pr_sp'] = []
        results[obj]['imgs_masks'] = []
        results[obj]['anomaly_maps'] = []
        results[obj]['color_anomaly_maps'] = []
        results[obj]['integrate_anomaly_maps'] = []

    for idx, items in enumerate(tqdm(test_dataloader)):
        image = items['img'].to(device)
        cls_name = items['cls_name']
        gt_mask = items['img_mask']
        gt_mask[gt_mask > 0.5], gt_mask[gt_mask <= 0.5] = 1, 0
        results[cls_name[0]]['imgs_masks'].append(gt_mask)
        results[cls_name[0]]['gt_sp'].extend(items['anomaly'].detach().cpu())
        render_image = items['d2_render_img'].to(device)
        b, nv, c, h, w = render_image.shape
        render_image = render_image.reshape(-1, c, h, w)
        d2_3d_cor = items['d2_3d_cor'].to(device)
        non_zero_index = items['non_zero_index'].to(device)
        non_zero_index = non_zero_index.unsqueeze(1).repeat(1, nv, 1, 1)
        whole_image = torch.cat([render_image, image], dim=0)

        with torch.no_grad():
            obj_type = cls_name[0]
            TextSlice.bind(clip, Clip_Text_Encoder, device)
            PTD = predefined_text_descriptions(obj_type)
            pos, neg = PTD["positive"], PTD["negative"]
            Visual_Embeddings, Hybrid_Visual_Embeddings = ACLIP_Image_Ecnoder.encode_image(whole_image, args.test_features_list, DPAM_layer=20)
            Hybrid_local_results = AnomalyCLIP_lib.compute_local_results(Hybrid_Visual_Embeddings, pos[:args.p0], neg[:args.n0], d2_3d_cor, non_zero_index, args)
            Local_results = AnomalyCLIP_lib.cmpl_local_results(items, Visual_Embeddings, pos[args.p0:args.p0 + args.p1], neg[args.n0:args.n0 + args.n1], args)
            Hybrid_global_results = AnomalyCLIP_lib.compute_global_results(Hybrid_Visual_Embeddings, pos[args.p0 + args.p1:], neg[args.n0 + args.n1:], d2_3d_cor, non_zero_index, args)
            Global_results = AnomalyCLIP_lib.cmpl_global_results(Visual_Embeddings, pos, neg, args)
            results[cls_name[0]]['integrate_pr_sp'].extend(AnomalyCLIP_lib.get_global_scores(Hybrid_global_results, Global_results, weights=args.test_weights))
            results[cls_name[0]]['integrate_anomaly_maps'].append(process_pixel_maps(Local_results, Hybrid_local_results, args))

    integrate_table_ls = []
    integrate_image_auroc_list = []
    integrate_image_ap_list = []
    integrate_pixel_auroc_list = []
    integrate_pixel_aupro_list = []
    obj_list = [c for c in test_data.obj_list]
    print('obj_list:', obj_list)
    for obj in obj_list:
        integrate_table = []
        integrate_table.append(obj)
        results[obj]['imgs_masks'] = torch.cat(results[obj]['imgs_masks'])
        results[obj]['integrate_anomaly_maps'] = torch.cat(results[obj]['integrate_anomaly_maps']).detach().cpu().numpy()
        integrate_image_auroc = image_level_metrics(results, obj, "image-auroc", modality='integrate_pr_sp')
        integrate_image_ap = image_level_metrics(results, obj, "image-ap", modality='integrate_pr_sp')
        integrate_pixel_auroc = pixel_level_metrics(results, obj, "pixel-auroc", modality='integrate_anomaly_maps')
        gt = results[obj]['imgs_masks']
        pr = results[obj]['integrate_anomaly_maps']
        gts = [gt[i, 0].cpu().numpy() if isinstance(gt, torch.Tensor) else gt[i, 0] for i in range(gt.shape[0])]
        predictions = [pr[i].cpu().numpy() if isinstance(pr, torch.Tensor) else pr[i] for i in range(pr.shape[0])]
        integrate_pixel_aupro, _ = calculate_au_pro(gts, predictions)
        integrate_table.append(str(np.round(integrate_pixel_auroc * 100, decimals=1)))
        integrate_table.append(str(np.round(integrate_pixel_aupro * 100, decimals=1)))
        integrate_table.append(str(np.round(integrate_image_auroc * 100, decimals=1)))
        integrate_table.append(str(np.round(integrate_image_ap * 100, decimals=1)))
        integrate_image_auroc_list.append(integrate_image_auroc)
        integrate_image_ap_list.append(integrate_image_ap)
        integrate_pixel_auroc_list.append(integrate_pixel_auroc)
        integrate_pixel_aupro_list.append(integrate_pixel_aupro)
        integrate_table_ls.append(integrate_table)

    integrate_table_ls.append(['mean', str(np.round(np.mean(integrate_pixel_auroc_list) * 100, decimals=1)),
                               str(np.round(np.mean(integrate_pixel_aupro_list) * 100, decimals=1)),
                               str(np.round(np.mean(integrate_image_auroc_list) * 100, decimals=1)),
                               str(np.round(np.mean(integrate_image_ap_list) * 100, decimals=1))])

    integrate_results = tabulate(integrate_table_ls,
                                 headers=['objects', 'pixel_auroc', 'pixel_aupro', 'image_auroc', 'image_ap'],
                                 tablefmt="pipe")
    print('integrate_results')
    logger.info("\n%s", integrate_results)

if __name__ == '__main__':
    parser = argparse.ArgumentParser("ZUMA", add_help=True)
    parser.add_argument("--config", type=str, required=True, help="yaml config path")
    parser.add_argument("--test_dataset", type=str, required=True, help="dataset name")
    parser.add_argument("--test_data_path", type=str, required=True, help="dataset root path")
    cmd = parser.parse_args()
    cfg = load_yaml_config(cmd.config)
    cfg["test_dataset"] = cmd.test_dataset
    cfg["test_data_path"] = cmd.test_data_path
    args = SimpleNamespace(**cfg)
    setup_seed(args.seed)
    test(args)