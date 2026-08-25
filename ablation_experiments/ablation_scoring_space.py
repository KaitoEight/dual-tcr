"""
Ablation study: SAM-space vs RADIO-space vs Dual-space scoring.
Compare TCR arbitration performance across different scoring latent spaces.
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
    parser.add_argument('--outdir', type=str, default='scoring_ablation')
    parser.add_argument('--ckpt', type=str, default='sam_vit_h_4b8939.pth')
    parser.add_argument('--ref_idx', type=str, default='00')
    parser.add_argument('--sam_type', type=str, default='vit_h')
    parser.add_argument('--scoring', type=str, default='sam', choices=['sam', 'radio', 'dual'])
    args = parser.parse_args()
    return args


def evaluate_mask_in_sam_space(mask, test_feat_sam, target_emb_sam):
    """Score mask in SAM feature space (decoder-aligned)."""
    _, h, w = test_feat_sam.shape
    mask_tensor = torch.from_numpy(mask).float().cuda().unsqueeze(0).unsqueeze(0)
    mask_resized = F.interpolate(mask_tensor, size=(h, w), mode="nearest").squeeze()
    mask_feat = test_feat_sam.permute(1, 2, 0)[mask_resized > 0]
    if mask_feat.shape[0] == 0:
        return -1.0
    mask_emb = mask_feat.mean(0).unsqueeze(0)
    mask_emb = mask_emb / mask_emb.norm(dim=-1, keepdim=True)
    target_emb_flat = target_emb_sam.squeeze()
    if len(target_emb_flat.shape) > 1:
        target_emb_flat = target_emb_flat[0]
    return (mask_emb @ target_emb_flat.unsqueeze(1)).item()


def evaluate_mask_in_radio_space(mask, test_feat_radio, target_emb_radio):
    """Score mask in RADIO feature space (semantic)."""
    H_r, W_r, C_r = test_feat_radio.shape
    mask_tensor = torch.from_numpy(mask).float().cuda().unsqueeze(0).unsqueeze(0)
    mask_resized = F.interpolate(mask_tensor, size=(H_r, W_r), mode="nearest").squeeze()
    mask_feat = test_feat_radio[mask_resized > 0]
    if mask_feat.shape[0] == 0:
        return -1.0
    mask_emb = mask_feat.mean(0).unsqueeze(0)
    mask_emb = mask_emb / mask_emb.norm(dim=-1, keepdim=True)
    target_emb_flat = target_emb_radio.squeeze()
    if len(target_emb_flat.shape) > 1:
        target_emb_flat = target_emb_flat[0]
    return (mask_emb @ target_emb_flat.unsqueeze(1)).item()


def compute_iou(mask1, mask2):
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    if union == 0:
        return 0.0
    return intersection / union


def process_category(obj_name, images_path, masks_path, predictor, radio, radio_transform, scoring):
    """Process one category with specified scoring method."""
    ref_image_path = os.path.join(images_path, obj_name, '00.jpg')
    ref_mask_path = os.path.join(masks_path, obj_name, '00.png')
    test_images_path = os.path.join(images_path, obj_name)
    test_masks_path = os.path.join(masks_path, obj_name)

    if not os.path.exists(ref_image_path) or not os.path.exists(ref_mask_path):
        return None, 0

    # Load reference
    ref_image = cv2.imread(ref_image_path)
    ref_image = cv2.cvtColor(ref_image, cv2.COLOR_BGR2RGB)
    ref_mask = cv2.imread(ref_mask_path, cv2.IMREAD_GRAYSCALE)
    ref_mask_binary = (ref_mask > 0).astype(np.float32)

    # Get reference features
    predictor.set_image(ref_image)
    ref_feat_sam = predictor.features.squeeze()

    ref_tensor = radio_transform(ref_image).unsqueeze(0).cuda()
    with torch.no_grad():
        _, spatial_features = radio(ref_tensor, feature_fmt='NCHW')
        ref_feat_radio = spatial_features.squeeze(0).permute(1, 2, 0)

    # Compute target embeddings
    ref_mask_tensor = torch.from_numpy(ref_mask_binary).float().cuda().unsqueeze(0).unsqueeze(0)

    # SAM target embedding
    _, H_s, W_s = ref_feat_sam.shape
    ref_mask_sam = F.interpolate(ref_mask_tensor, size=(H_s, W_s), mode="nearest").squeeze()
    target_feat_sam = ref_feat_sam.permute(1, 2, 0)[ref_mask_sam > 0]
    if target_feat_sam.shape[0] == 0:
        return None, 0
    target_emb_sam = target_feat_sam.mean(0).unsqueeze(0)
    target_emb_sam = target_emb_sam / target_emb_sam.norm(dim=-1, keepdim=True)

    # RADIO target embedding
    H_r, W_r, C_r = ref_feat_radio.shape
    ref_mask_rad = F.interpolate(ref_mask_tensor, size=(H_r, W_r), mode="nearest").squeeze()
    target_feat_rad = ref_feat_radio[ref_mask_rad > 0]
    if target_feat_rad.shape[0] == 0:
        return None, 0
    target_emb_radio = target_feat_rad.mean(0).unsqueeze(0)
    target_emb_radio = target_emb_radio / target_emb_radio.norm(dim=-1, keepdim=True)

    # Process test images
    test_images = sorted([f for f in os.listdir(test_images_path) if f.endswith('.jpg') and f != '00.jpg'])
    ious = []

    for test_img_file in test_images:
        test_idx = test_img_file.replace('.jpg', '')
        test_image_path = os.path.join(test_images_path, test_img_file)
        test_mask_path = os.path.join(test_masks_path, test_idx + '.png')

        if not os.path.exists(test_mask_path):
            continue

        test_image = cv2.imread(test_image_path)
        test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)
        gt_mask = cv2.imread(test_mask_path, cv2.IMREAD_GRAYSCALE)
        gt_binary = (gt_mask > 0).astype(np.uint8)

        # SAM features for test image
        predictor.set_image(test_image)
        test_feat_sam = predictor.features.squeeze()

        # RADIO features for test image
        test_tensor = radio_transform(test_image).unsqueeze(0).cuda()
        with torch.no_grad():
            _, spatial_features = radio(test_tensor, feature_fmt='NCHW')
            test_feat_radio = spatial_features.squeeze(0).permute(1, 2, 0)

        # Get positive prompt from SAM similarity
        # test_feat_sam: [256, H, W], target_emb_sam: [1, 256]
        test_feat_sam_flat = test_feat_sam.flatten(1).t()  # [H*W, 256]
        target_emb_sam_flat = target_emb_sam.squeeze().unsqueeze(0)  # [1, 256]
        sim_sam_flat = torch.nn.functional.cosine_similarity(
            test_feat_sam_flat, target_emb_sam_flat, dim=1
        )  # [H*W]
        sim_sam = sim_sam_flat.reshape(test_feat_sam.shape[1], test_feat_sam.shape[2])
        sim_sam_up = F.interpolate(sim_sam.unsqueeze(0).unsqueeze(0), size=test_image.shape[:2], mode='bilinear').squeeze()
        max_loc = np.unravel_index(sim_sam_up.cpu().numpy().argmax(), sim_sam_up.shape)
        p_pos_sam = np.array([max_loc[0], max_loc[1]])

        # SAM prediction
        masks_sam, scores_sam, _, _ = predictor.predict(
            np.array([p_pos_sam]), np.array([1]), multimask_output=True
        )
        best_idx_sam = np.argmax(scores_sam)
        mask_sam = masks_sam[best_idx_sam].astype(np.uint8)

        # Get positive prompt from RADIO similarity
        # test_feat_radio: [H, W, C], target_emb_radio: [1, C]
        test_feat_radio_flat = test_feat_radio.flatten(0, 1)  # [H*W, C]
        target_emb_radio_flat = target_emb_radio.squeeze().unsqueeze(0)  # [1, C]
        sim_rad_flat = torch.nn.functional.cosine_similarity(
            test_feat_radio_flat, target_emb_radio_flat, dim=1
        )  # [H*W]
        sim_rad = sim_rad_flat.reshape(test_feat_radio.shape[0], test_feat_radio.shape[1])
        sim_rad_up = F.interpolate(sim_rad.unsqueeze(0).unsqueeze(0), size=test_image.shape[:2], mode='bilinear').squeeze()
        max_loc_r = np.unravel_index(sim_rad_up.cpu().numpy().argmax(), sim_rad_up.shape)
        p_pos_rad = np.array([max_loc_r[0], max_loc_r[1]])

        # RADIO-guided SAM prediction
        masks_rad, scores_rad, _, _ = predictor.predict(
            np.array([p_pos_rad]), np.array([1]), multimask_output=True
        )
        best_idx_rad = np.argmax(scores_rad)
        mask_radio = masks_rad[best_idx_rad].astype(np.uint8)

        # Evaluate using specified scoring space
        if scoring == 'sam':
            s_sam = evaluate_mask_in_sam_space(mask_sam, test_feat_sam, target_emb_sam)
            s_rad = evaluate_mask_in_sam_space(mask_radio, test_feat_sam, target_emb_sam)
        elif scoring == 'radio':
            s_sam = evaluate_mask_in_radio_space(mask_sam, test_feat_radio, target_emb_radio)
            s_rad = evaluate_mask_in_radio_space(mask_radio, test_feat_radio, target_emb_radio)
        else:  # dual
            s_sam_s = evaluate_mask_in_sam_space(mask_sam, test_feat_sam, target_emb_sam)
            s_rad_s = evaluate_mask_in_sam_space(mask_radio, test_feat_sam, target_emb_sam)
            s_sam_r = evaluate_mask_in_radio_space(mask_sam, test_feat_radio, target_emb_radio)
            s_rad_r = evaluate_mask_in_radio_space(mask_radio, test_feat_radio, target_emb_radio)
            s_sam = 0.5 * s_sam_s + 0.5 * s_sam_r
            s_rad = 0.5 * s_rad_s + 0.5 * s_rad_r

        # Select winner
        if s_rad > s_sam:
            winning_mask = mask_radio
        else:
            winning_mask = mask_sam

        iou = compute_iou(winning_mask, gt_binary)
        ious.append(iou)

    return np.mean(ious) * 100 if ious else None, len(ious)


def main(scoring='sam'):
    print(f"\n{'='*60}")
    print(f"Testing {scoring.upper()}-space scoring")
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
    print("SCORING SPACE ABLATION RESULTS")
    print("="*60)
    for scoring, miou in results.items():
        print(f"  {scoring.upper():>6}-space: {miou:>6.2f}% mIoU")

    with open('scoring_ablation_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("\nResults saved to scoring_ablation_results.json")
