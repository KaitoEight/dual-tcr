"""
Script đánh giá quantitative localization metrics cho 3 encoders trên PerSeg dataset.

Metrics:
1. Prompt Hit Rate (%): Tỷ lệ positive prompt p+ nằm trong GT mask (pixel = 1)
2. Normalized Target Distance: Khoảng cách từ p+ đến centroid GT mask, chia cho diagonal của GT BB

Encoders được so sánh:
- DINOv2 (ViT-L/14)
- SAM (ViT-H)
- RADIO (ViT-L/16)
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
import timm
from scipy.spatial.distance import cdist

from per_segment_anything import sam_model_registry, SamPredictor


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='./data')
    parser.add_argument('--outdir', type=str, default='eval_prompt_localization')
    parser.add_argument('--ckpt', type=str, default='sam_vit_h_4b8939.pth')
    parser.add_argument('--ref_idx', type=str, default='00')
    parser.add_argument('--sam_type', type=str, default='vit_h')
    parser.add_argument('--save_csv', type=str, default='prompt_localization_results.csv')
    args = parser.parse_args()
    return args


def point_selection(mask_sim, topk=1):
    """Chọn positive point (điểm có similarity cao nhất)
    Returns: array of shape (topk, 2) with [y, x] format
    """
    w, h = mask_sim.shape

    # Positive point: vị trí có giá trị cao nhất
    topk_xy = mask_sim.flatten(0).topk(topk)[1]  # indices flattened
    topk_x = (topk_xy // h).unsqueeze(0)  # column (width)
    topk_y = (topk_xy - topk_x * h)  # row (height)
    topk_xy = torch.cat((topk_y, topk_x), dim=0).permute(1, 0)  # [y, x]
    topk_xy = topk_xy.cpu().numpy()

    return topk_xy


def compute_bb_diagonal(gt_mask):
    """Tính độ dài đường chéo của bounding box GT"""
    rows = np.any(gt_mask, axis=1)
    cols = np.any(gt_mask, axis=0)

    if not np.any(rows) or not np.any(cols):
        return 1.0  # Tránh chia cho 0

    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    # Đường chéo của BB
    diagonal = np.sqrt((rmax - rmin)**2 + (cmax - cmin)**2)
    return max(diagonal, 1.0)


def compute_smoothness_metric(sim_map):
    """
    Đo spatial smoothness/continuity của similarity map.
    Grid artifacts tạo ra các vùng có giá trị "răng cưa" (checkerboard pattern).

    Phương pháp: Tính gradient magnitude trung bình
    - Map mượt mà → gradient thấp
    - Map bị grid artifacts → gradient cao (nhiễu)
    """
    # Compute spatial gradients
    dy = np.diff(sim_map, axis=0)
    dx = np.diff(sim_map, axis=1)

    # Gradient magnitude
    grad_mag = np.sqrt(np.mean(dy**2) + np.mean(dx**2))

    return grad_mag


def compute_spatial_coherence(sim_map, threshold=0.7):
    """
    Đo spatial coherence: tỷ lệ pixels có giá trị > threshold mà KHÔNG bị ngắt quãng.

    Grid artifacts làm cho các vùng high-similarity bị chia cắt thành nhiều mảnh nhỏ
    thay vì tạo thành một vùng liên tục (blob).
    """
    binary = (sim_map > threshold).astype(np.uint8)

    # Tìm connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)

    if num_labels <= 1:
        return 0.0  # Không có foreground

    # Tính total foreground pixels
    total_fg = binary.sum()

    # Lấy kích thước của largest component
    if len(stats) > 1:
        largest_component_size = max(stats[1:, cv2.CC_STAT_AREA])
        coherence = largest_component_size / total_fg if total_fg > 0 else 0.0
    else:
        coherence = 0.0

    return coherence


def compute_foreground_spread(sim_map, percentile=90):
    """
    Đo spread (lan tỏa) của các điểm có similarity cao.

    Grid artifacts làm cho vùng high-similarity bị phân tán thành nhiều điểm rải rác
    thay vì tập trung thành một blob.
    """
    # Lấy các điểm top-k percentile
    threshold = np.percentile(sim_map, percentile)
    high_sim_mask = (sim_map >= threshold).astype(np.uint8)

    # Tìm bounding box của các điểm high-similarity
    ys, xs = np.where(high_sim_mask > 0)

    if len(ys) < 5:
        return float('inf')

    # Tính spread: diện tích BB / số pixels
    bb_area = (ys.max() - ys.min() + 1) * (xs.max() - xs.min() + 1)
    pixel_count = len(ys)

    # Spread ratio: BB lớn nhưng pixel count nhỏ = phân tán
    # BB nhỏ và pixel count lớn = tập trung
    spread = bb_area / pixel_count if pixel_count > 0 else float('inf')

    return spread


def compute_centroid(gt_mask):
    """Tính centroid của GT mask"""
    ys, xs = np.where(gt_mask > 0)
    if len(ys) == 0:
        return np.array([0, 0])
    return np.array([ys.mean(), xs.mean()])


def evaluate_single_image(ref_image, ref_mask, test_image, test_mask, encoder_type,
                         predictor, dinov2_model, radio_model,
                         dino_transform, radio_transform):
    """
    Đánh giá localization cho một cặp ảnh reference-test.
    Trả về: (hit_rate, normalized_distance)
    """
    h_img, w_img = test_image.shape[:2]

    # Chuyển test_mask sang grayscale nếu cần
    if len(test_mask.shape) == 3:
        test_mask_gray = cv2.cvtColor(test_mask, cv2.COLOR_BGR2GRAY)
    else:
        test_mask_gray = test_mask

    # Binary mask - mask values can be 38 or 255 depending on the dataset
    gt_binary = (test_mask_gray > 0).astype(np.uint8)

    # Khởi tạo kết quả
    results = {
        'dino_hit': False,
        'dino_norm_dist': float('inf'),
        'dino_smoothness': float('inf'),
        'dino_coherence': 0.0,
        'dino_spread': float('inf'),
        'sam_hit': False,
        'sam_norm_dist': float('inf'),
        'sam_smoothness': float('inf'),
        'sam_coherence': 0.0,
        'sam_spread': float('inf'),
        'radio_hit': False,
        'radio_norm_dist': float('inf'),
        'radio_smoothness': float('inf'),
        'radio_coherence': 0.0,
        'radio_spread': float('inf'),
    }

    # ==================== DINOv2 ====================
    if encoder_type in ['dino', 'all']:
        with torch.no_grad():
            # Feature extraction from reference
            ref_tensor = dino_transform(ref_image).unsqueeze(0).cuda()
            ref_feat = dinov2_model.forward_features(ref_tensor)[:, 1:, :]
            B, N, C = ref_feat.shape
            H_dino = W_dino = int(np.sqrt(N))
            ref_feat = ref_feat.reshape(B, H_dino, W_dino, C).squeeze(0)

            # Target embedding từ ref mask (NOT test mask)
            ref_mask_tensor = torch.from_numpy(cv2.cvtColor(ref_mask, cv2.COLOR_BGR2GRAY) if len(ref_mask.shape) == 3 else ref_mask).float().cuda().unsqueeze(0).unsqueeze(0)
            ref_mask_dino = F.interpolate(ref_mask_tensor, size=(H_dino, W_dino), mode="nearest").squeeze()

            target_feat = ref_feat[ref_mask_dino > 0]
            if target_feat.shape[0] == 0:
                target_emb = ref_feat[H_dino//2, W_dino//2, :].unsqueeze(0)
            else:
                target_emb = target_feat.mean(0).unsqueeze(0)
            target_emb = target_emb / target_emb.norm(dim=-1, keepdim=True)

            # Test image features
            test_tensor = dino_transform(test_image).unsqueeze(0).cuda()
            test_feat = dinov2_model.forward_features(test_tensor)[:, 1:, :]
            test_feat = test_feat.reshape(1, H_dino, W_dino, C).squeeze(0)
            test_feat_norm = test_feat / test_feat.norm(dim=-1, keepdim=True)

            # Similarity map
            sim_flat = target_emb @ test_feat_norm.reshape(H_dino * W_dino, C).t()
            sim = sim_flat.reshape(H_dino, W_dino)

            # Resize về kích thước ảnh gốc
            sim_resized = F.interpolate(
                sim.unsqueeze(0).unsqueeze(0),
                size=(h_img, w_img),
                mode="bilinear"
            ).squeeze()

            # Positive point selection
            p_pos = point_selection(sim_resized, topk=1)
            p_pos = p_pos[0]  # [y, x]

            # Compute metrics using TEST mask GT
            if p_pos[0] < gt_binary.shape[0] and p_pos[1] < gt_binary.shape[1]:
                results['dino_hit'] = gt_binary[int(p_pos[0]), int(p_pos[1])] == 1

            centroid = compute_centroid(gt_binary)
            diagonal = compute_bb_diagonal(gt_binary)
            dist = np.sqrt((p_pos[0] - centroid[0])**2 + (p_pos[1] - centroid[1])**2)
            results['dino_norm_dist'] = dist / diagonal

            # Spatial quality metrics (using sim at feature resolution)
            sim_np = sim.cpu().numpy()
            results['dino_smoothness'] = compute_smoothness_metric(sim_np)
            results['dino_coherence'] = compute_spatial_coherence(sim_np, threshold=0.7)
            results['dino_spread'] = compute_foreground_spread(sim_np, percentile=90)

    # ==================== SAM ====================
    if encoder_type in ['sam', 'all']:
        # Reference image embedding
        predictor.set_image(ref_image)
        ref_feat_sam = predictor.features.squeeze()  # [256, 64, 64]
        C_sam, h_sam, w_sam = ref_feat_sam.shape

        # Resize ref_mask (NOT test_mask) to match feature size [64, 64]
        ref_mask_np = cv2.cvtColor(ref_mask, cv2.COLOR_BGR2GRAY) if len(ref_mask.shape) == 3 else ref_mask
        ref_mask_tensor = torch.from_numpy(ref_mask_np).float().cuda().unsqueeze(0).unsqueeze(0)
        ref_mask_sam = F.interpolate(ref_mask_tensor, size=(h_sam, w_sam), mode="nearest").squeeze()

        # Get target embedding from reference
        ref_feat_sam_perm = ref_feat_sam.permute(1, 2, 0)  # [64, 64, 256]
        target_feat = ref_feat_sam_perm[ref_mask_sam > 0]
        if target_feat.shape[0] == 0:
            target_emb = ref_feat_sam_perm[h_sam//2, w_sam//2, :].unsqueeze(0)
        else:
            target_emb = target_feat.mean(0).unsqueeze(0)
        target_emb = target_emb / target_emb.norm(dim=-1, keepdim=True)

        # Test image features
        predictor.set_image(test_image)
        test_feat_sam = predictor.features.squeeze()
        test_feat_sam_perm = test_feat_sam.permute(1, 2, 0)  # [64, 64, 256]
        test_feat_flat = test_feat_sam_perm.reshape(-1, C_sam)
        test_feat_flat = test_feat_flat / test_feat_flat.norm(dim=-1, keepdim=True)

        # Similarity map
        sim_flat = (target_emb @ test_feat_flat.t()).squeeze(0)
        sim = sim_flat.reshape(h_sam, w_sam)
        sim_resized = F.interpolate(
            sim.unsqueeze(0).unsqueeze(0),
            size=(h_img, w_img),
            mode="bilinear"
        ).squeeze()

        # Positive point selection
        p_pos = point_selection(sim_resized, topk=1)
        p_pos = p_pos[0]

        # Compute metrics using TEST mask GT
        if p_pos[0] < gt_binary.shape[0] and p_pos[1] < gt_binary.shape[1]:
            results['sam_hit'] = gt_binary[int(p_pos[0]), int(p_pos[1])] == 1

        centroid = compute_centroid(gt_binary)
        diagonal = compute_bb_diagonal(gt_binary)

        # Spatial quality metrics (using sim at feature resolution)
        sim_np = sim.cpu().numpy()
        results['sam_smoothness'] = compute_smoothness_metric(sim_np)
        results['sam_coherence'] = compute_spatial_coherence(sim_np, threshold=0.7)
        results['sam_spread'] = compute_foreground_spread(sim_np, percentile=90)
        dist = np.sqrt((p_pos[0] - centroid[0])**2 + (p_pos[1] - centroid[1])**2)
        results['sam_norm_dist'] = dist / diagonal

    # ==================== RADIO ====================
    if encoder_type in ['radio', 'all']:
        with torch.no_grad():
            # Reference feature
            ref_tensor = radio_transform(ref_image).unsqueeze(0).cuda()
            _, ref_feat = radio_model(ref_tensor, feature_fmt='NCHW')
            ref_feat = ref_feat.squeeze(0).permute(1, 2, 0)  # [H, W, C]
            C_rad, H_rad, W_rad = ref_feat.shape[2], ref_feat.shape[0], ref_feat.shape[1]

            # Convert ref_mask to grayscale for target embedding extraction
            ref_mask_np = cv2.cvtColor(ref_mask, cv2.COLOR_BGR2GRAY) if len(ref_mask.shape) == 3 else ref_mask
            ref_mask_tensor = torch.from_numpy(ref_mask_np).float().cuda().unsqueeze(0).unsqueeze(0)
            ref_mask_resized = F.interpolate(ref_mask_tensor, size=(H_rad, W_rad), mode="nearest").squeeze()

            target_feat = ref_feat[ref_mask_resized > 0]
            if target_feat.shape[0] == 0:
                target_emb = ref_feat[H_rad//2, W_rad//2, :].unsqueeze(0)
            else:
                target_emb = target_feat.mean(0).unsqueeze(0)
            target_emb = target_emb / target_emb.norm(dim=-1, keepdim=True)

            # Test image feature
            test_tensor = radio_transform(test_image).unsqueeze(0).cuda()
            _, test_feat = radio_model(test_tensor, feature_fmt='NCHW')
            test_feat = test_feat.squeeze(0).permute(1, 2, 0)
            test_feat_norm = test_feat / test_feat.norm(dim=-1, keepdim=True)

            sim_flat = target_emb @ test_feat_norm.reshape(H_rad * W_rad, C_rad).t()
            sim = sim_flat.reshape(H_rad, W_rad)
            sim_resized = F.interpolate(
                sim.unsqueeze(0).unsqueeze(0),
                size=(h_img, w_img),
                mode="bilinear"
            ).squeeze()

            # Positive point selection
            p_pos = point_selection(sim_resized, topk=1)
            p_pos = p_pos[0]

            # Compute metrics (gt_binary already defined at function start)
            if p_pos[0] < gt_binary.shape[0] and p_pos[1] < gt_binary.shape[1]:
                results['radio_hit'] = gt_binary[int(p_pos[0]), int(p_pos[1])] == 1

            centroid = compute_centroid(gt_binary)
            diagonal = compute_bb_diagonal(gt_binary)
            dist = np.sqrt((p_pos[0] - centroid[0])**2 + (p_pos[1] - centroid[1])**2)
            results['radio_norm_dist'] = dist / diagonal

            # Spatial quality metrics (using sim at feature resolution)
            sim_np = sim.cpu().numpy()
            results['radio_smoothness'] = compute_smoothness_metric(sim_np)
            results['radio_coherence'] = compute_spatial_coherence(sim_np, threshold=0.7)
            results['radio_spread'] = compute_foreground_spread(sim_np, percentile=90)

    return results


def main():
    args = get_arguments()
    print("Arguments:", args)

    images_path = args.data + '/Images/'
    masks_path = args.data + '/Annotations/'

    print("======> Loading models...")

    # Load SAM
    sam = sam_model_registry[args.sam_type](checkpoint=args.ckpt).cuda()
    predictor = SamPredictor(sam)

    # Load DINOv2 (ViT-L/14) via timm
    dinov2 = timm.create_model('vit_large_patch14_dinov2.lvd142m', pretrained=True).cuda()
    dinov2.eval()

    # Load RADIO
    radio = torch.hub.load('NVlabs/RADIO', 'radio_model', version='radio_v2.5-l', progress=False).cuda()
    radio.eval()

    # Transforms
    dino_transform = T.Compose([
        T.ToPILImage(),
        T.Resize((518, 518)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    radio_transform = T.Compose([
        T.ToPILImage(),
        T.Resize((512, 512)),
        T.ToTensor(),
    ])

    print("======> Starting evaluation on PerSeg...")

    # Lưu kết quả chi tiết
    all_results = []

    # Đếm số categories và images
    categories = sorted([d for d in os.listdir(images_path) if not d.startswith('.')])
    print(f"Found {len(categories)} categories")

    for obj_name in tqdm(categories, desc="Categories"):
        obj_images_path = os.path.join(images_path, obj_name)
        obj_masks_path = os.path.join(masks_path, obj_name)

        # Load reference image và mask
        ref_image_path = os.path.join(obj_images_path, args.ref_idx + '.jpg')
        ref_mask_path = os.path.join(obj_masks_path, args.ref_idx + '.png')

        if not os.path.exists(ref_image_path) or not os.path.exists(ref_mask_path):
            print(f"Warning: Missing files for {obj_name}")
            continue

        ref_image = cv2.imread(ref_image_path)
        ref_image = cv2.cvtColor(ref_image, cv2.COLOR_BGR2RGB)
        ref_mask = cv2.imread(ref_mask_path)
        ref_mask = cv2.cvtColor(ref_mask, cv2.COLOR_BGR2RGB)

        # Đánh giá trên tất cả ảnh test (bao gồm cả ảnh ref để so sánh)
        test_images = sorted([f for f in os.listdir(obj_images_path) if f.endswith('.jpg')])

        for test_img_file in test_images:
            test_idx = test_img_file.replace('.jpg', '')
            test_image_path = os.path.join(obj_images_path, test_img_file)
            test_mask_path = os.path.join(obj_masks_path, test_idx + '.png')

            test_image = cv2.imread(test_image_path)
            test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)

            # Load test mask if exists (for metrics computation)
            if os.path.exists(test_mask_path):
                test_mask = cv2.imread(test_mask_path)
                test_mask = cv2.cvtColor(test_mask, cv2.COLOR_BGR2RGB)
            else:
                # If no test mask, skip this image for evaluation
                continue

            results = evaluate_single_image(
                ref_image, ref_mask, test_image, test_mask,
                encoder_type='all',
                predictor=predictor,
                dinov2_model=dinov2,
                radio_model=radio,
                dino_transform=dino_transform,
                radio_transform=radio_transform
            )

            # Lưu kết quả
            result_row = {
                'category': obj_name,
                'image_idx': test_idx,
                'dino_hit': results['dino_hit'],
                'dino_norm_dist': results['dino_norm_dist'],
                'dino_smoothness': results['dino_smoothness'],
                'dino_coherence': results['dino_coherence'],
                'dino_spread': results['dino_spread'],
                'sam_hit': results['sam_hit'],
                'sam_norm_dist': results['sam_norm_dist'],
                'sam_smoothness': results['sam_smoothness'],
                'sam_coherence': results['sam_coherence'],
                'sam_spread': results['sam_spread'],
                'radio_hit': results['radio_hit'],
                'radio_norm_dist': results['radio_norm_dist'],
                'radio_smoothness': results['radio_smoothness'],
                'radio_coherence': results['radio_coherence'],
                'radio_spread': results['radio_spread'],
            }
            all_results.append(result_row)

    # ==================== Tổng hợp kết quả ====================
    print("\n" + "="*60)
    print("PROMPT LOCALIZATION EVALUATION RESULTS")
    print("="*60)

    # Tính toán thống kê
    n_total = len(all_results)

    dino_hits = sum(1 for r in all_results if r['dino_hit'])
    sam_hits = sum(1 for r in all_results if r['sam_hit'])
    radio_hits = sum(1 for r in all_results if r['radio_hit'])

    dino_hit_rate = dino_hits / n_total * 100
    sam_hit_rate = sam_hits / n_total * 100
    radio_hit_rate = radio_hits / n_total * 100

    dino_dists = [r['dino_norm_dist'] for r in all_results if r['dino_norm_dist'] != float('inf')]
    sam_dists = [r['sam_norm_dist'] for r in all_results if r['sam_norm_dist'] != float('inf')]
    radio_dists = [r['radio_norm_dist'] for r in all_results if r['radio_norm_dist'] != float('inf')]

    dino_mean_dist = np.mean(dino_dists) if dino_dists else float('inf')
    sam_mean_dist = np.mean(sam_dists) if sam_dists else float('inf')
    radio_mean_dist = np.mean(radio_dists) if radio_dists else float('inf')

    dino_std_dist = np.std(dino_dists) if dino_dists else float('inf')
    sam_std_dist = np.std(sam_dists) if sam_dists else float('inf')
    radio_std_dist = np.std(radio_dists) if radio_dists else float('inf')

    # Spatial quality metrics
    dino_smoothness = [r['dino_smoothness'] for r in all_results if r['dino_smoothness'] != float('inf')]
    sam_smoothness = [r['sam_smoothness'] for r in all_results if r['sam_smoothness'] != float('inf')]
    radio_smoothness = [r['radio_smoothness'] for r in all_results if r['radio_smoothness'] != float('inf')]

    dino_coherence = [r['dino_coherence'] for r in all_results if r['dino_coherence'] > 0]
    sam_coherence = [r['sam_coherence'] for r in all_results if r['sam_coherence'] > 0]
    radio_coherence = [r['radio_coherence'] for r in all_results if r['radio_coherence'] > 0]

    dino_spread = [r['dino_spread'] for r in all_results if r['dino_spread'] != float('inf')]
    sam_spread = [r['sam_spread'] for r in all_results if r['sam_spread'] != float('inf')]
    radio_spread = [r['radio_spread'] for r in all_results if r['radio_spread'] != float('inf')]

    print(f"\nTotal images evaluated: {n_total}")
    print(f"\n--- Metric 1: Prompt Hit Rate (%) ---")
    print(f"  DINOv2 (ViT-L/14): {dino_hit_rate:.2f}% ({dino_hits}/{n_total})")
    print(f"  SAM   (ViT-H):     {sam_hit_rate:.2f}% ({sam_hits}/{n_total})")
    print(f"  RADIO (ViT-L/16):  {radio_hit_rate:.2f}% ({radio_hits}/{n_total})")

    print(f"\n--- Metric 2: Normalized Target Distance ---")
    print(f"  DINOv2 (ViT-L/14): {dino_mean_dist:.4f} ± {dino_std_dist:.4f}")
    print(f"  SAM   (ViT-H):     {sam_mean_dist:.4f} ± {sam_std_dist:.4f}")
    print(f"  RADIO (ViT-L/16):  {radio_mean_dist:.4f} ± {radio_std_dist:.4f}")

    print(f"\n--- Metric 3: Spatial Smoothness (Gradient Magnitude - LOWER is BETTER) ---")
    print(f"  DINOv2 (ViT-L/14): {np.mean(dino_smoothness):.6f} ± {np.std(dino_smoothness):.6f}")
    print(f"  SAM   (ViT-H):     {np.mean(sam_smoothness):.6f} ± {np.std(sam_smoothness):.6f}")
    print(f"  RADIO (ViT-L/16):  {np.mean(radio_smoothness):.6f} ± {np.std(radio_smoothness):.6f}")

    print(f"\n--- Metric 4: Spatial Coherence (Higher is BETTER - 1.0 = perfect blob) ---")
    print(f"  DINOv2 (ViT-L/14): {np.mean(dino_coherence):.4f} ± {np.std(dino_coherence):.4f}")
    print(f"  SAM   (ViT-H):     {np.mean(sam_coherence):.4f} ± {np.std(sam_coherence):.4f}")
    print(f"  RADIO (ViT-L/16):  {np.mean(radio_coherence):.4f} ± {np.std(radio_coherence):.4f}")

    print(f"\n--- Metric 5: Foreground Spread (Lower = more concentrated) ---")
    print(f"  DINOv2 (ViT-L/14): {np.mean(dino_spread):.2f} ± {np.std(dino_spread):.2f}")
    print(f"  SAM   (ViT-H):     {np.mean(sam_spread):.2f} ± {np.std(sam_spread):.2f}")
    print(f"  RADIO (ViT-L/16):  {np.mean(radio_spread):.2f} ± {np.std(radio_spread):.2f}")

    # Lưu kết quả ra CSV
    import csv
    with open(args.save_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
        writer.writeheader()
        writer.writerows(all_results)

    print(f"\nDetailed results saved to: {args.save_csv}")

    # Trả về dictionary để có thể sử dụng trong script khác
    return {
        'n_total': n_total,
        'dino_hit_rate': dino_hit_rate,
        'sam_hit_rate': sam_hit_rate,
        'radio_hit_rate': radio_hit_rate,
        'dino_mean_dist': dino_mean_dist,
        'sam_mean_dist': sam_mean_dist,
        'radio_mean_dist': radio_mean_dist,
        'dino_std_dist': dino_std_dist,
        'sam_std_dist': sam_std_dist,
        'radio_std_dist': radio_std_dist,
        'dino_smoothness': np.mean(dino_smoothness),
        'sam_smoothness': np.mean(sam_smoothness),
        'radio_smoothness': np.mean(radio_smoothness),
        'dino_coherence': np.mean(dino_coherence),
        'sam_coherence': np.mean(sam_coherence),
        'radio_coherence': np.mean(radio_coherence),
        'dino_spread': np.mean(dino_spread),
        'sam_spread': np.mean(sam_spread),
        'radio_spread': np.mean(radio_spread),
    }


if __name__ == "__main__":
    results = main()
