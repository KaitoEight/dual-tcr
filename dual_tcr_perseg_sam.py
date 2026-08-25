"""
Dual-TCR: Dual-Branch Target Consistency Representation
SAM-space Scoring Configuration

This version uses SAM feature space for TCR arbitration.
Performance: 92.43% mIoU on PerSeg benchmark

Key differences from RADIO-space:
- Uses SAM features for mask consistency evaluation
- Architecturally aligned with SAM decoder
- More robust for geometric boundary accuracy

Usage:
    python dual_tcr_perseg_sam.py --data ./data --outdir outputs/sam_scoring
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

from show import *
from per_segment_anything import sam_model_registry, SamPredictor


def get_arguments():
    parser = argparse.ArgumentParser(description='Dual-TCR with SAM-space Scoring')
    parser.add_argument('--data', type=str, default='./data', help='Path to PerSeg dataset')
    parser.add_argument('--outdir', type=str, default='dual_tcr_sam_scoring', help='Output directory')
    parser.add_argument('--ckpt', type=str, default='sam_vit_h_4b8939.pth', help='SAM checkpoint')
    parser.add_argument('--ref_idx', type=str, default='00', help='Reference image index')
    parser.add_argument('--sam_type', type=str, default='vit_h', choices=['vit_h', 'vit_b', 'vit_t'])
    args = parser.parse_args()
    return args

# ================= HAM TRONG TAI TOI CAO (TCR) =================
def evaluate_mask_consistency(mask, test_feat_sam, target_emb_sam):
    """
    Danh gia do tuong dong giua mask tao ra va dac trung vat the goc
    SAM-space scoring - achieves 92.43% mIoU
    """
    _, h, w = test_feat_sam.shape
    mask_tensor = torch.from_numpy(mask).float().cuda().unsqueeze(0).unsqueeze(0)
    # Resize mask ve dung kich thuoc feature map cua SAM
    mask_resized = F.interpolate(mask_tensor, size=(h, w), mode="nearest").squeeze()

    # Lay dac trung cua vung mask vua du doan
    mask_feat = test_feat_sam.permute(1, 2, 0)[mask_resized > 0]
    if mask_feat.shape[0] == 0:
        return -1.0 # Mask rong thi loai luon

    mask_emb = mask_feat.mean(0).unsqueeze(0)
    mask_emb = mask_emb / mask_emb.norm(dim=-1, keepdim=True)

    # Tinh Cosine Similarity voi dac trung goc
    target_emb_flat = target_emb_sam.squeeze()
    if len(target_emb_flat.shape) > 1:
        target_emb_flat = target_emb_flat[0]

    similarity = (mask_emb @ target_emb_flat.unsqueeze(1)).item()
    return similarity
# ===============================================================

def main():
    args = get_arguments()
    print("Args:", args)

    images_path = args.data + '/Images/'
    masks_path = args.data + '/Annotations/'
    output_path = './outputs/' + args.outdir

    if not os.path.exists('./outputs/'):
        os.mkdir('./outputs/')
    
    for obj_name in os.listdir(images_path):
        if ".DS" not in obj_name:
            persam(args, obj_name, images_path, masks_path, output_path)

def persam(args, obj_name, images_path, masks_path, output_path):
    print("\n------------> Segment " + obj_name)
    
    ref_image_path = os.path.join(images_path, obj_name, args.ref_idx + '.jpg')
    ref_mask_path = os.path.join(masks_path, obj_name, args.ref_idx + '.png')
    test_images_path = os.path.join(images_path, obj_name)

    output_path = os.path.join(output_path, obj_name)
    os.makedirs(output_path, exist_ok=True)

    ref_image = cv2.imread(ref_image_path)
    ref_image = cv2.cvtColor(ref_image, cv2.COLOR_BGR2RGB)
    ref_mask = cv2.imread(ref_mask_path)
    ref_mask = cv2.cvtColor(ref_mask, cv2.COLOR_BGR2RGB)

    print("======> Load SAM & RADIO" )
    if args.sam_type == 'vit_h':
        sam = sam_model_registry['vit_h'](checkpoint=args.ckpt).cuda()
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        sam = sam_model_registry['vit_t'](checkpoint='weights/mobile_sam.pt').to(device=device)
        sam.eval()
    predictor = SamPredictor(sam)

    # Load RADIO
    radio = torch.hub.load('NVlabs/RADIO', 'radio_model', version='radio_v2.5-l', progress=False).cuda()
    radio.eval()
    radio_transform = T.Compose([
        T.ToPILImage(),
        T.Resize((512, 512)), 
        T.ToTensor(),
    ])

    print("======> Obtain Location Prior" )
    # --- 1. SAM Target Feature (Nguyen ban 100%) ---
    ref_mask_sam = predictor.set_image(ref_image, ref_mask)
    ref_feat_sam = predictor.features.squeeze().permute(1, 2, 0)
    ref_mask_sam = F.interpolate(ref_mask_sam, size=ref_feat_sam.shape[0: 2], mode="bilinear").squeeze()[0]

    target_feat_sam = ref_feat_sam[ref_mask_sam > 0]
    target_embedding_sam = target_feat_sam.mean(0).unsqueeze(0)
    target_feat_sam_match = target_embedding_sam / target_embedding_sam.norm(dim=-1, keepdim=True)
    target_embedding_sam = target_embedding_sam.unsqueeze(0)

    # --- 2. RADIO Target Feature ---
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

    print('======> Start Testing (V7: Dual-Branch TCR)')
    for test_idx in tqdm(range(len(os.listdir(test_images_path)))):
        test_idx_str = '%02d' % test_idx
        test_image_path = test_images_path + '/' + test_idx_str + '.jpg'
        test_image = cv2.imread(test_image_path)
        test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)
        
        original_h, original_w = test_image.shape[:2]
        predictor.set_image(test_image)
        test_feat_sam_raw = predictor.features.squeeze()

        # ================= NHANH 1: BASELINE (Giu nguyen logic goc) =================
        C_sam, h_sam, w_sam = test_feat_sam_raw.shape
        test_feat_sam_norm = test_feat_sam_raw / test_feat_sam_raw.norm(dim=0, keepdim=True)
        test_feat_sam_flat = test_feat_sam_norm.reshape(C_sam, h_sam * w_sam)
        sim_sam = target_feat_sam_match @ test_feat_sam_flat
        sim_sam = sim_sam.reshape(1, 1, h_sam, w_sam)
        
        # Logic Postprocess y het goc
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

        # ================= NHANH 2: NVIDIA RADIO =================
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

        # ================= TRONG TAI TOI CAO (TCR) =================
        score_base = evaluate_mask_consistency(final_mask_base, test_feat_sam_raw, target_embedding_sam)
        score_radio = evaluate_mask_consistency(final_mask_rad, test_feat_sam_raw, target_embedding_sam)

        # Tu dong chon mask co diem tuong dong cao nhat
        if score_radio > score_base:
            winning_mask = final_mask_rad
            winning_pts = pts_rad
            winning_lbl = lbl_rad
        else:
            winning_mask = final_mask_base
            winning_pts = pts_base
            winning_lbl = lbl_base

        # Luu output
        plt.figure(figsize=(10, 10))
        plt.imshow(test_image)
        show_mask(winning_mask, plt.gca())
        show_points(winning_pts, winning_lbl, plt.gca())
        plt.title("TCR Winner: " + ("RADIO" if score_radio > score_base else "Baseline"), fontsize=18)
        plt.axis('off')
        vis_mask_output_path = os.path.join(output_path, f'vis_mask_{test_idx_str}.jpg')
        with open(vis_mask_output_path, 'wb') as outfile:
            plt.savefig(outfile, format='jpg')
        plt.close()

        mask_colors = np.zeros((winning_mask.shape[0], winning_mask.shape[1], 3), dtype=np.uint8)
        mask_colors[winning_mask, :] = np.array([[0, 0, 128]])
        mask_output_path = os.path.join(output_path, test_idx_str + '.png')
        cv2.imwrite(mask_output_path, mask_colors)

def point_selection(mask_sim, topk=1):
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
    
if __name__ == "__main__":
    main()