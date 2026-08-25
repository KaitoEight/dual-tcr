"""
Ablation study for τ (tau) parameter in HRM safety rollback.
Tests τ values from 0.0 to 0.1 with step 0.01 on PerSeg dataset.

Expected: Peak mIoU at τ = 0.02, then flat or slight decrease.
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
import matplotlib.pyplot as plt

from per_segment_anything import sam_model_registry, SamPredictor


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='./data')
    parser.add_argument('--outdir_prefix', type=str, default='ablation_tau')
    parser.add_argument('--ckpt', type=str, default='sam_vit_h_4b8939.pth')
    parser.add_argument('--ref_idx', type=str, default='00')
    parser.add_argument('--sam_type', type=str, default='vit_h')
    parser.add_argument('--tau_values', type=str, default='0.0,0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1')
    args = parser.parse_args()
    return args


def point_selection(mask_sim, topk=1):
    """Top-1 and Top-last point selection"""
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


def compute_iou(pred, gt):
    intersection = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    if union == 0:
        return 0.0
    return intersection / union


def evaluate_mask_consistency(mask, test_feat_sam, target_emb_sam):
    """Compute TCR score for a mask in SAM feature space."""
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


def hybrid_refining_with_tau(predictor, mask_a, mask_b, logits_a, logits_b,
                             pts_a, lbl_a, pts_b, lbl_b,
                             score_a, score_b, target_emb_sam, tau):
    """
    Hybrid Refining Module with τ safety rollback.

    Formula from paper:
    M* = M_hybrid if score_hybrid >= score_w - τ else M_w
    """
    # 1. Determine winner/loser
    used_b = score_b > score_a
    if used_b:
        win_mask, lose_mask = mask_b, mask_a
        win_logits, lose_logits = logits_b, logits_a
        win_pts, win_lbl = pts_b, lbl_b
        lose_pts, lose_lbl = pts_a, lbl_a
        win_score, lose_score = score_b, score_a
    else:
        win_mask, lose_mask = mask_a, mask_b
        win_logits, lose_logits = logits_a, logits_b
        win_pts, win_lbl = pts_a, lbl_a
        lose_pts, lose_lbl = pts_b, lbl_b
        win_score, lose_score = score_a, score_b

    H_img, W_img = win_mask.shape

    # 2. Agreement region
    agreement = np.logical_and(win_mask, lose_mask)

    # 3. Combined point prompts
    win_pos = win_pts[win_lbl == 1] if len(win_pts) > 0 else np.empty((0, 2))
    win_neg = win_pts[win_lbl == 0] if len(win_pts) > 0 else np.empty((0, 2))
    lose_pos = lose_pts[lose_lbl == 1] if len(lose_pts) > 0 else np.empty((0, 2))
    lose_neg = lose_pts[lose_lbl == 0] if len(lose_pts) > 0 else np.empty((0, 2))

    # Filter loser positives to agreement region
    extra_pos_list = []
    for pt in lose_pos:
        x_p, y_p = int(round(pt[0])), int(round(pt[1]))
        if 0 <= y_p < H_img and 0 <= x_p < W_img:
            if agreement[y_p, x_p]:
                extra_pos_list.append(pt)

    pts_parts = [win_pos]
    lbl_parts = [np.ones(len(win_pos), dtype=np.int64)]

    if len(extra_pos_list) > 0:
        extra_pos = np.array(extra_pos_list)
        pts_parts.append(extra_pos)
        lbl_parts.append(np.ones(len(extra_pos), dtype=np.int64))

    pts_parts.append(win_neg)
    lbl_parts.append(np.zeros(len(win_neg), dtype=np.int64))
    pts_parts.append(lose_neg)
    lbl_parts.append(np.zeros(len(lose_neg), dtype=np.int64))

    combined_pts = np.concatenate(pts_parts, axis=0).astype(np.float32)
    combined_lbl = np.concatenate(lbl_parts, axis=0).astype(np.int64)

    # 4. Weighted logits fusion (gate: lose_score > 0.3 * win_score)
    eps = 1e-6
    w_win = (win_score + eps) / (win_score + lose_score + 2 * eps)
    w_lose = 1.0 - w_win

    if lose_score > 0.3 * win_score and lose_score > 0:
        fused_logits = w_win * win_logits + w_lose * lose_logits
    else:
        fused_logits = win_logits

    # 5. Bounding box (gate: lose_score >= 0.5 * win_score)
    y_w, x_w = np.nonzero(win_mask)
    if len(y_w) == 0 or len(x_w) == 0:
        return win_mask, win_score

    if lose_score >= 0.5 * win_score:
        y_l, x_l = np.nonzero(lose_mask)
        if len(y_l) > 0 and len(x_l) > 0:
            ys = np.concatenate([y_w, y_l])
            xs = np.concatenate([x_w, x_l])
        else:
            ys, xs = y_w, x_w
    else:
        ys, xs = y_w, x_w

    box = np.array([xs.min(), ys.min(), xs.max(), ys.max()])

    # 6. Call SAM decoder for hybrid mask
    try:
        masks_final, scores_final, _, _ = predictor.predict(
            point_coords=combined_pts,
            point_labels=combined_lbl,
            box=box[None, :],
            mask_input=fused_logits[0:1, :, :],
            multimask_output=True,
        )
        hybrid_mask = masks_final[np.argmax(scores_final)]
        hybrid_score = evaluate_mask_consistency(hybrid_mask, predictor.features.squeeze(), target_emb_sam)
    except Exception as e:
        print(f"[HRM] Error: {e}, fallback to winner mask")
        hybrid_mask = win_mask
        hybrid_score = win_score

    # 7. τ safety rollback
    if hybrid_score >= win_score - tau:
        return hybrid_mask, hybrid_score
    else:
        return win_mask, win_score


def process_single_tau(tau, args, images_path, masks_path, predictor, radio, radio_transform):
    """Process all images for a single τ value and return mIoU."""
    output_path = f'./outputs/{args.outdir_prefix}_tau_{tau:.2f}'
    os.makedirs(output_path, exist_ok=True)

    categories = sorted([d for d in os.listdir(images_path) if not d.startswith('.')])
    all_ious = []

    for obj_name in tqdm(categories, desc=f"τ={tau:.2f}"):
        obj_images_path = os.path.join(images_path, obj_name)
        obj_masks_path = os.path.join(masks_path, obj_name)
        obj_output_path = os.path.join(output_path, obj_name)
        os.makedirs(obj_output_path, exist_ok=True)

        ref_image_path = os.path.join(obj_images_path, args.ref_idx + '.jpg')
        ref_mask_path = os.path.join(obj_masks_path, args.ref_idx + '.png')

        if not os.path.exists(ref_image_path) or not os.path.exists(ref_mask_path):
            continue

        ref_image = cv2.imread(ref_image_path)
        ref_image = cv2.cvtColor(ref_image, cv2.COLOR_BGR2RGB)
        ref_mask = cv2.imread(ref_mask_path)
        ref_mask = cv2.cvtColor(ref_mask, cv2.COLOR_BGR2RGB)

        # SAM Branch
        predictor.set_image(ref_image)
        ref_feat_sam = predictor.features.squeeze()
        ref_mask_tensor = torch.from_numpy(ref_mask[:,:,0]).float().unsqueeze(0).unsqueeze(0).cuda()
        ref_mask_sam = F.interpolate(ref_mask_tensor, size=ref_feat_sam.shape[-2:], mode="nearest").squeeze()
        target_emb_sam = ref_feat_sam[:, ref_mask_sam > 0].mean(dim=1)
        target_emb_sam = target_emb_sam / target_emb_sam.norm(dim=-1, keepdim=True)

        # RADIO Branch
        with torch.no_grad():
            ref_tensor = radio_transform(ref_image).unsqueeze(0).cuda()
            _, ref_feat_radio = radio(ref_tensor, feature_fmt='NCHW')
            ref_feat_radio = ref_feat_radio.squeeze(0).permute(1, 2, 0)
            C_rad, H_rad, W_rad = ref_feat_radio.shape[2], ref_feat_radio.shape[0], ref_feat_radio.shape[1]
            ref_mask_radio = F.interpolate(ref_mask_tensor, size=(H_rad, W_rad), mode="nearest").squeeze()
            target_emb_radio = ref_feat_radio[ref_mask_radio > 0].mean(dim=0).unsqueeze(0)
            target_emb_radio = target_emb_radio / target_emb_radio.norm(dim=-1, keepdim=True)

        # Process test images
        test_images = [f for f in os.listdir(obj_images_path) if f.endswith('.jpg')]
        for test_img_file in test_images:
            test_idx = test_img_file.replace('.jpg', '')
            test_image_path = os.path.join(obj_images_path, test_img_file)
            gt_mask_path = os.path.join(obj_masks_path, test_idx + '.png')

            if not os.path.exists(gt_mask_path):
                continue

            test_image = cv2.imread(test_image_path)
            test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)

            gt_mask = cv2.imread(gt_mask_path)
            gt_mask = cv2.cvtColor(gt_mask, cv2.COLOR_BGR2GRAY) > 0

            # SAM Branch prediction
            predictor.set_image(test_image)
            test_feat_sam = predictor.features.squeeze()
            C_sam, h_sam, w_sam = test_feat_sam.shape
            test_feat_sam_norm = test_feat_sam / test_feat_sam.norm(dim=0, keepdim=True)
            test_feat_sam_flat = test_feat_sam_norm.reshape(C_sam, -1)
            sim_sam_flat = torch.matmul(target_emb_sam, test_feat_sam_flat)
            sim_sam_map = sim_sam_flat.reshape(h_sam, w_sam).unsqueeze(0).unsqueeze(0)
            sim_sam_up = F.interpolate(sim_sam_map, size=predictor.input_size, mode='bilinear')
            sim_sam_up = predictor.model.postprocess_masks(sim_sam_up, predictor.input_size, predictor.original_size).squeeze()

            pts_sam_pos, lbl_sam_pos, pts_sam_neg, lbl_sam_neg = point_selection(sim_sam_up, topk=1)
            pts_sam = np.concatenate([pts_sam_pos, pts_sam_neg], axis=0)
            lbl_sam = np.concatenate([lbl_sam_pos, lbl_sam_neg], axis=0)

            masks_sam, scores_sam, logits_sam, _ = predictor.predict(pts_sam, lbl_sam, multimask_output=True)
            best_idx_sam = np.argmax(scores_sam)
            mask_sam = masks_sam[best_idx_sam]
            # logits from SAM predictor is already numpy array
            logits_sam_best = logits_sam[best_idx_sam:best_idx_sam+1] if isinstance(logits_sam, np.ndarray) else logits_sam[best_idx_sam:best_idx_sam+1].cpu().numpy()

            # RADIO Branch prediction
            with torch.no_grad():
                test_tensor = radio_transform(test_image).unsqueeze(0).cuda()
                _, test_feat_radio = radio(test_tensor, feature_fmt='NCHW')
                test_feat_radio = test_feat_radio.squeeze(0)  # [C, H, W]
                C_rad_t, H_t, W_t = test_feat_radio.shape
                test_feat_radio_norm = test_feat_radio / test_feat_radio.norm(dim=0, keepdim=True)
                sim_radio_flat = torch.matmul(target_emb_radio.squeeze(0), test_feat_radio_norm.reshape(C_rad_t, -1))
                sim_radio_map = sim_radio_flat.reshape(1, 1, H_t, W_t)
                sim_radio_up = F.interpolate(sim_radio_map, size=predictor.input_size, mode='bilinear').squeeze()

            pts_r_pos, lbl_r_pos, pts_r_neg, lbl_r_neg = point_selection(sim_radio_up, topk=1)
            pts_radio = np.concatenate([pts_r_pos, pts_r_neg], axis=0)
            lbl_radio = np.concatenate([lbl_r_pos, lbl_r_neg], axis=0)

            masks_radio, scores_radio, logits_radio, _ = predictor.predict(pts_radio, lbl_radio, multimask_output=True)
            best_idx_radio = np.argmax(scores_radio)
            mask_radio = masks_radio[best_idx_radio]
            logits_radio_best = logits_radio[best_idx_radio:best_idx_radio+1] if isinstance(logits_radio, np.ndarray) else logits_radio[best_idx_radio:best_idx_radio+1].cpu().numpy()

            # TCR Scores (in SAM feature space)
            s_sam = evaluate_mask_consistency(mask_sam.astype(np.uint8), test_feat_sam, target_emb_sam.unsqueeze(0))
            s_radio = evaluate_mask_consistency(mask_radio.astype(np.uint8), test_feat_sam, target_emb_sam.unsqueeze(0))

            # Apply Hybrid Refining with τ
            final_mask, final_score = hybrid_refining_with_tau(
                predictor, mask_sam, mask_radio,
                logits_sam_best, logits_radio_best,
                pts_sam, lbl_sam, pts_radio, lbl_radio,
                s_sam, s_radio, target_emb_sam.unsqueeze(0), tau
            )

            # Compute IoU
            iou = compute_iou(final_mask, gt_mask)
            all_ious.append(iou)

            # Save mask
            mask_colors = np.zeros((final_mask.shape[0], final_mask.shape[1], 3), dtype=np.uint8)
            mask_colors[final_mask, :] = np.array([[0, 0, 128]])
            cv2.imwrite(os.path.join(obj_output_path, test_idx + '.png'), mask_colors)

    return np.mean(all_ious) * 100, len(all_ious)


def main():
    args = get_arguments()
    print("Args:", args)

    images_path = args.data + '/Images/'
    masks_path = args.data + '/Annotations/'

    # Parse tau values
    tau_values = [float(t) for t in args.tau_values.split(',')]
    print(f"Testing τ values: {tau_values}")

    # Load models
    print("\n======> Loading SAM ======")
    sam = sam_model_registry[args.sam_type](checkpoint=args.ckpt).cuda()
    sam.eval()
    predictor = SamPredictor(sam)

    print("======> Loading RADIO ======")
    radio = torch.hub.load('NVlabs/RADIO', 'radio_model', version='radio_v2.5-l', progress=False).cuda()
    radio.eval()

    radio_transform = T.Compose([
        T.ToPILImage(),
        T.Resize((512, 512)),
        T.ToTensor(),
    ])
    print("======> Models Loaded ======\n")

    # Run ablation
    results = {}
    for tau in tau_values:
        print(f"\n{'='*60}")
        print(f"Running ablation for τ = {tau:.2f}")
        print(f"{'='*60}")
        miou, n_images = process_single_tau(tau, args, images_path, masks_path, predictor, radio, radio_transform)
        results[tau] = miou
        print(f"τ = {tau:.2f}: mIoU = {miou:.2f}% ({n_images} images)")

    # Print summary table
    print("\n" + "="*60)
    print("ABLATION RESULTS: HRM τ Parameter")
    print("="*60)
    print(f"{'τ':>6} | {'mIoU (%)':>10} | {'Δ from τ=0.02':>15}")
    print("-" * 40)
    best_tau = max(results, key=results.get)
    for tau in sorted(results.keys()):
        delta = results[tau] - results[0.02]
        marker = " ← Best" if tau == best_tau else ""
        print(f"{tau:>6.2f} | {results[tau]:>10.2f} | {delta:>+15.2f}{marker}")

    print(f"\nOptimal τ = {best_tau:.2f} with mIoU = {results[best_tau]:.2f}%")

    # Save results
    import json
    results_file = f'./outputs/{args.outdir_prefix}_results.json'
    with open(results_file, 'w') as f:
        json.dump({str(k): v for k, v in results.items()}, f, indent=2)
    print(f"\nResults saved to {results_file}")


if __name__ == '__main__':
    main()
