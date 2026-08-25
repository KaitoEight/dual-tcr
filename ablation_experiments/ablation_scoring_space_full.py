"""
Ablation study: SAM-space vs RADIO-space vs Dual-space scoring.
Based on persam_v7tcr.py with full Dual-TCR pipeline.
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
import torch
from torch.nn import functional as F
import cv2
from tqdm import tqdm
import argparse
import warnings
warnings.filterwarnings('ignore')
import torchvision.transforms as T
import matplotlib
matplotlib.use('Agg')

from per_segment_anything import sam_model_registry, SamPredictor


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='./data')
    parser.add_argument('--outdir', type=str, default='scoring_ablation_full')
    parser.add_argument('--ckpt', type=str, default='sam_vit_h_4b8939.pth')
    parser.add_argument('--ref_idx', type=str, default='00')
    parser.add_argument('--sam_type', type=str, default='vit_h')
    parser.add_argument('--scoring', type=str, default='sam', choices=['sam', 'radio', 'dual'])
    args = parser.parse_args()
    return args


def evaluate_mask_in_sam_space(mask, test_feat_sam_perm, target_emb_sam):
    """Score mask in SAM feature space - measures geometric quality.
    test_feat_sam_perm: [H, W, C] tensor on GPU
    target_emb_sam: [1, 1, C] tensor on GPU
    """
    H, W, C = test_feat_sam_perm.shape
    mask_np = np.array(mask).astype(np.float32)
    mask_tensor = torch.from_numpy(mask_np).float().cuda().unsqueeze(0).unsqueeze(0)
    mask_resized = F.interpolate(mask_tensor, size=(H, W), mode="nearest").squeeze()
    mask_feat = test_feat_sam_perm[mask_resized > 0]
    if mask_feat.shape[0] == 0:
        return -1.0
    mask_emb = mask_feat.mean(0).unsqueeze(0)
    mask_emb = mask_emb / mask_emb.norm(dim=-1, keepdim=True)
    # target_emb_sam is [1, 1, C], squeeze to [C]
    target_emb_flat = target_emb_sam.squeeze()  # [1, 1, C] -> [C] or stays [1, 1, C]
    if len(target_emb_flat.shape) == 2:
        target_emb_flat = target_emb_flat.squeeze(0).squeeze(0)  # [1, C] -> [C]
    elif len(target_emb_flat.shape) == 1:
        target_emb_flat = target_emb_flat  # [C]
    return (mask_emb.squeeze(0) @ target_emb_flat).item()


def evaluate_mask_in_radio_space(mask, test_feat_radio, target_emb_radio):
    """Score mask in RADIO feature space - measures semantic coverage.
    test_feat_radio: [H, W, C] tensor on GPU
    target_emb_radio: [1, C] tensor on GPU
    """
    H_r, W_r, C_r = test_feat_radio.shape
    mask_np = np.array(mask).astype(np.float32)
    mask_tensor = torch.from_numpy(mask_np).float().cuda().unsqueeze(0).unsqueeze(0)
    mask_resized = F.interpolate(mask_tensor, size=(H_r, W_r), mode="nearest").squeeze()
    mask_feat = test_feat_radio[mask_resized > 0]
    if mask_feat.shape[0] == 0:
        return -1.0
    mask_emb = mask_feat.mean(0).unsqueeze(0)
    mask_emb = mask_emb / mask_emb.norm(dim=-1, keepdim=True)
    # target_emb_radio is [1, C]
    target_emb_flat = target_emb_radio.squeeze(0)  # [1, C] -> [C]
    return (mask_emb.squeeze(0) @ target_emb_flat).item()


def compute_iou(mask1, mask2):
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    if union == 0:
        return 0.0
    return intersection / union


def point_selection(mask_sim, topk=1):
    """Select positive and negative points from similarity map."""
    w, h = mask_sim.shape
    topk_xy = mask_sim.flatten(0).topk(topk)[1]
    topk_x = (topk_xy // h).unsqueeze(0)
    topk_y = (topk_xy - topk_x * h)
    topk_xy = torch.cat((topk_y, topk_x), dim=0).permute(1, 0)
    topk_label = np.array([1] * topk)
    topk_xy = topk_xy.cpu().numpy()

    last_xy = mask_sim.flatten(0).topk(topk, largest=False)[1]
    last_x = (last_xy // h).unsqueeze(0)
    last_y = (last_xy - last_x * h)
    last_xy = torch.cat((last_y, last_x), dim=0).permute(1, 0)
    last_label = np.array([0] * topk)
    last_xy = last_xy.cpu().numpy()

    return topk_xy, topk_label, last_xy, last_label


def process_category(obj_name, images_path, masks_path, predictor, radio, radio_transform, scoring):
    """Process one category with full Dual-TCR pipeline."""
    ref_image_path = os.path.join(images_path, obj_name, '00.jpg')
    ref_mask_path = os.path.join(masks_path, obj_name, '00.png')
    test_images_path = os.path.join(images_path, obj_name)
    test_masks_path = os.path.join(masks_path, obj_name)

    if not os.path.exists(ref_image_path) or not os.path.exists(ref_mask_path):
        return None, 0

    ref_image = cv2.imread(ref_image_path)
    ref_image = cv2.cvtColor(ref_image, cv2.COLOR_BGR2RGB)
    ref_mask = cv2.imread(ref_mask_path)
    ref_mask = cv2.cvtColor(ref_mask, cv2.COLOR_BGR2RGB)

    # SAM Target Feature
    ref_mask_sam = predictor.set_image(ref_image, ref_mask)
    ref_feat_sam = predictor.features.squeeze().permute(1, 2, 0)
    ref_mask_sam = F.interpolate(ref_mask_sam, size=ref_feat_sam.shape[0:2], mode="bilinear").squeeze()[0]

    target_feat_sam = ref_feat_sam[ref_mask_sam > 0]
    target_embedding_sam = target_feat_sam.mean(0).unsqueeze(0)
    target_feat_sam_match = target_embedding_sam / target_embedding_sam.norm(dim=-1, keepdim=True)
    target_embedding_sam = target_embedding_sam.unsqueeze(0)

    # RADIO Target Feature
    with torch.no_grad():
        img_tensor = radio_transform(ref_image).unsqueeze(0).cuda()
        _, spatial_features = radio(img_tensor, feature_fmt='NCHW')
        ref_feat_radio = spatial_features.squeeze(0).permute(1, 2, 0)
        H_r, W_r, C_r = ref_feat_radio.shape

        ref_mask_tensor = torch.from_numpy(ref_mask[:,:,0]).float().cuda().unsqueeze(0).unsqueeze(0)
        ref_mask_radio = F.interpolate(ref_mask_tensor, size=(H_r, W_r), mode="nearest").squeeze()

        target_feat_fg_radio = ref_feat_radio[ref_mask_radio > 0]
        if target_feat_fg_radio.shape[0] == 0:
             target_emb_radio = ref_feat_radio[H_r//2, W_r//2, :].unsqueeze(0)
        else:
             target_emb_radio = target_feat_fg_radio.mean(0).unsqueeze(0)
        target_emb_radio = target_emb_radio / target_emb_radio.norm(dim=-1, keepdim=True)

    # Process test images
    ious = []
    test_files = [f for f in os.listdir(test_images_path) if f.endswith('.jpg') and f != '00.jpg']

    for test_idx_str in sorted([f.replace('.jpg', '') for f in test_files])[:5]:  # Limit for speed
        test_image_path = os.path.join(test_images_path, test_idx_str + '.jpg')
        test_mask_path = os.path.join(test_masks_path, test_idx_str + '.png')

        if not os.path.exists(test_mask_path):
            continue

        test_image = cv2.imread(test_image_path)
        test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)
        gt_mask = cv2.imread(test_mask_path, cv2.IMREAD_GRAYSCALE)
        gt_binary = (gt_mask > 0).astype(np.uint8)

        original_h, original_w = test_image.shape[:2]
        predictor.set_image(test_image)
        test_feat_sam_raw = predictor.features.squeeze()

        # Baseline SAM prediction
        C_sam, h_sam, w_sam = test_feat_sam_raw.shape
        test_feat_sam_norm = test_feat_sam_raw / test_feat_sam_raw.norm(dim=0, keepdim=True)
        test_feat_sam_flat = test_feat_sam_norm.reshape(C_sam, h_sam * w_sam)
        sim_sam = target_feat_sam_match @ test_feat_sam_flat
        sim_sam = sim_sam.reshape(1, 1, h_sam, w_sam)

        sim_sam_post = F.interpolate(sim_sam, scale_factor=4, mode="bilinear")
        sim_sam_post = predictor.model.postprocess_masks(sim_sam_post, input_size=predictor.input_size, original_size=predictor.original_size).squeeze()

        topk_xy_base, topk_label_base, last_xy_base, last_label_base = point_selection(sim_sam_post, topk=1)
        pts_base = np.concatenate([topk_xy_base, last_xy_base], axis=0)
        lbl_base = np.concatenate([topk_label_base, last_label_base], axis=0)

        sim_attn_base = (sim_sam_post - sim_sam_post.mean()) / torch.std(sim_sam_post)
        sim_attn_base = F.interpolate(sim_attn_base.unsqueeze(0).unsqueeze(0), size=(64, 64), mode="bilinear")
        attn_sim_base = sim_attn_base.sigmoid_().unsqueeze(0).flatten(3)

        masks_base, scores_base, logits_base, _ = predictor.predict(
            point_coords=pts_base, point_labels=lbl_base, multimask_output=False,
            attn_sim=attn_sim_base, target_embedding=target_embedding_sam
        )
        masks_base, scores_base, logits_base, _ = predictor.predict(
            point_coords=pts_base, point_labels=lbl_base, mask_input=logits_base[0:1,:,:], multimask_output=True)
        best_idx_base = np.argmax(scores_base)

        y, x = np.nonzero(masks_base[best_idx_base])
        if len(y)>0 and len(x)>0:
            box_base = np.array([x.min(), y.min(), x.max(), y.max()])
            masks_base, scores_base, _, _ = predictor.predict(
                point_coords=pts_base, point_labels=lbl_base, box=box_base[None,:],
                mask_input=logits_base[best_idx_base:best_idx_base+1,:,:], multimask_output=True)
        final_mask_base = masks_base[np.argmax(scores_base)]

        # RADIO-guided prediction
        with torch.no_grad():
            test_img_tensor = radio_transform(test_image).unsqueeze(0).cuda()
            _, spatial_features = radio(test_img_tensor, feature_fmt='NCHW')
            test_feat_radio = spatial_features.squeeze(0).permute(1, 2, 0)
            test_feat_radio_norm = test_feat_radio / test_feat_radio.norm(dim=-1, keepdim=True)
            test_feat_radio_flat = test_feat_radio_norm.reshape(H_r * W_r, C_r).t()

            sim_radio = target_emb_radio @ test_feat_radio_flat
            sim_radio = sim_radio.reshape(1, 1, H_r, W_r)

        sim_radio_orig = F.interpolate(sim_radio, size=(original_h, original_w), mode="bilinear").squeeze()

        topk_xy_rad, topk_label_rad, last_xy_rad, last_label_rad = point_selection(sim_radio_orig, topk=1)
        pts_rad = np.concatenate([topk_xy_rad, last_xy_rad], axis=0)
        lbl_rad = np.concatenate([topk_label_rad, last_label_rad], axis=0)

        sim_attn_rad = (sim_radio_orig - sim_radio_orig.mean()) / torch.std(sim_radio_orig)
        sim_attn_rad = F.interpolate(sim_attn_rad.unsqueeze(0).unsqueeze(0), size=(64, 64), mode="bilinear")
        attn_sim_rad = sim_attn_rad.sigmoid_().unsqueeze(0).flatten(3)

        masks_rad, scores_rad, logits_rad, _ = predictor.predict(
            point_coords=pts_rad, point_labels=lbl_rad, multimask_output=False,
            attn_sim=attn_sim_rad, target_embedding=target_embedding_sam
        )
        masks_rad, scores_rad, logits_rad, _ = predictor.predict(
            point_coords=pts_rad, point_labels=lbl_rad, mask_input=logits_rad[0:1,:,:], multimask_output=True)
        best_idx_rad = np.argmax(scores_rad)

        y, x = np.nonzero(masks_rad[best_idx_rad])
        if len(y)>0 and len(x)>0:
            box_rad = np.array([x.min(), y.min(), x.max(), y.max()])
            masks_rad, scores_rad, _, _ = predictor.predict(
                point_coords=pts_rad, point_labels=lbl_rad, box=box_rad[None,:],
                mask_input=logits_rad[best_idx_rad:best_idx_rad+1,:,:], multimask_output=True)
        final_mask_rad = masks_rad[np.argmax(scores_rad)]

        # Score using specified scoring space
        if scoring == 'sam':
            s_base = evaluate_mask_in_sam_space(final_mask_base, test_feat_sam_raw.permute(1, 2, 0), target_embedding_sam)
            s_rad = evaluate_mask_in_sam_space(final_mask_rad, test_feat_sam_raw.permute(1, 2, 0), target_embedding_sam)
        elif scoring == 'radio':
            s_base = evaluate_mask_in_radio_space(final_mask_base, test_feat_radio, target_emb_radio)
            s_rad = evaluate_mask_in_radio_space(final_mask_rad, test_feat_radio, target_emb_radio)
        else:  # dual
            s_base_sam = evaluate_mask_in_sam_space(final_mask_base, test_feat_sam_raw.permute(1, 2, 0), target_embedding_sam)
            s_rad_sam = evaluate_mask_in_sam_space(final_mask_rad, test_feat_sam_raw.permute(1, 2, 0), target_embedding_sam)
            s_base_rad = evaluate_mask_in_radio_space(final_mask_base, test_feat_radio, target_emb_radio)
            s_rad_rad = evaluate_mask_in_radio_space(final_mask_rad, test_feat_radio, target_emb_radio)
            s_base = 0.5 * s_base_sam + 0.5 * s_base_rad
            s_rad = 0.5 * s_rad_sam + 0.5 * s_rad_rad

        # Select winner
        if s_rad > s_base:
            winning_mask = final_mask_rad
        else:
            winning_mask = final_mask_base

        iou = compute_iou(winning_mask, gt_binary)
        ious.append(iou)

    return np.mean(ious) * 100 if ious else None, len(ious)


def main(scoring='sam'):
    print(f"\n{'='*60}")
    print(f"Testing {scoring.upper()}-space scoring (FULL DUAL-TCR PIPELINE)")
    print(f"{'='*60}")

    args = get_arguments()
    args.scoring = scoring

    images_path = args.data + '/Images/'
    masks_path = args.data + '/Annotations/'

    # Load SAM
    print("======> Loading SAM ======")
    sam = sam_model_registry[args.sam_type](checkpoint=args.ckpt).cuda()
    sam.eval()
    predictor = SamPredictor(sam)

    # Load RADIO
    print("======> Loading RADIO ======")
    radio = torch.hub.load('NVlabs/RADIO', 'radio_model', version='radio_v2.5-l', progress=False).cuda()
    radio.eval()

    radio_transform = T.Compose([
        T.ToPILImage(),
        T.Resize((512, 512)),
        T.ToTensor(),
    ])
    print("======> Models Loaded ======\n")

    # Process all categories
    categories = sorted([d for d in os.listdir(images_path) if not d.startswith('.') and not d.endswith('.DS_Store')])
    all_ious = []

    for obj_name in tqdm(categories, desc=f"Scoring: {scoring}"):
        iou, n = process_category(obj_name, images_path, masks_path, predictor, radio, radio_transform, scoring)
        if iou is not None:
            all_ious.append(iou)

    final_miou = np.mean(all_ious) if all_ious else 0.0
    print(f"\n{scoring.upper()}-space Scoring: mIoU = {final_miou:.2f}% ({len(all_ious)} categories)")
    return final_miou


if __name__ == '__main__':
    import json

    results = {}
    for scoring in ['sam', 'radio', 'dual']:
        miou = main(scoring)
        results[scoring] = miou

    print("\n" + "="*60)
    print("SCORING SPACE ABLATION RESULTS (FULL PIPELINE)")
    print("="*60)
    for scoring, miou in results.items():
        print(f"  {scoring.upper():>6}-space: {miou:>6.2f}% mIoU")

    with open('scoring_ablation_full_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("\nResults saved to scoring_ablation_full_results.json")
