"""
Point tracking utilities used in DragGAN.

- SimpleRAFTTracker: Your original feature-based tracker (100% unchanged core logic + add anti-wander clamp)
- PIPsTracker: Algorithm-level feature dimension supplement (no model modification)
   Pre-trained weights load normally (model structure unchanged)
   Fix 8-frame requirement: Use cubic spline interpolation (ref -> curr) for stability
   Better than frame repetition or linear interpolation
   100% non-invasive to PIPS source code
   Add Nearest/SimpleRAFT robust logic to solve non-convergence issue
"""
import torch
import torch.nn.functional as F
import numpy as np
import sys
import os
import importlib.util
from typing import List, Optional
import traceback

# =============================================================================
# Frame Interpolation Methods (Multiple strategies for 8-frame sequence)
# =============================================================================

def interpolate_frames_cubic_spline(img_ref: torch.Tensor, img_curr: torch.Tensor) -> List[torch.Tensor]:
    """
    Cubic Hermite spline interpolation (RECOMMENDED - most stable)
    
    Advantages:
    - Smooth, continuous derivatives (C1 continuity)
    - No over-oscillation (unlike cubic polynomials)
    - Stable for image interpolation
    
    Returns: 8 frames [ref, f1, f2, f3, f4, f5, f6, curr]
    """
    frames_list = [img_ref]
    for i in range(1, 7):
        t = float(i) / 7.0
        # Cubic Hermite basis functions (zero velocity endpoints = stable)
        h00 = 2.0 * t**3 - 3.0 * t**2 + 1.0
        h01 = -2.0 * t**3 + 3.0 * t**2
        interp_frame = h00 * img_ref + h01 * img_curr
        frames_list.append(interp_frame)
    frames_list.append(img_curr)
    return frames_list


def interpolate_frames_easing(img_ref: torch.Tensor, img_curr: torch.Tensor, easing_type: str = 'ease_in_out') -> List[torch.Tensor]:
    """
    Easing function interpolation
    
    Easing types:
    - 'linear': Constant velocity
    - 'ease_in': Accelerate from start
    - 'ease_out': Decelerate to end
    - 'ease_in_out': Accelerate then decelerate
    
    Returns: 8 frames [ref, f1, f2, f3, f4, f5, f6, curr]
    """
    frames_list = [img_ref]
    for i in range(1, 7):
        t = float(i) / 7.0
        
        if easing_type == 'linear':
            alpha = t
        elif easing_type == 'ease_in':
            alpha = t ** 2
        elif easing_type == 'ease_out':
            alpha = 1.0 - (1.0 - t) ** 2
        elif easing_type == 'ease_in_out':  # Most natural
            alpha = 3.0 * t**2 - 2.0 * t**3 if t < 0.5 else 1.0 - 2.0 * (1.0 - t)**2
        else:
            alpha = t
        
        interp_frame = (1.0 - alpha) * img_ref + alpha * img_curr
        frames_list.append(interp_frame)
    frames_list.append(img_curr)
    return frames_list


def interpolate_frames_quadratic(img_ref: torch.Tensor, img_curr: torch.Tensor) -> List[torch.Tensor]:
    """
    Quadratic interpolation
    
    Advantages:
    - Smoother than linear
    - Lighter computation than cubic
    - Good for real-time applications
    
    Returns: 8 frames [ref, f1, f2, f3, f4, f5, f6, curr]
    """
    frames_list = [img_ref]
    for i in range(1, 7):
        t = float(i) / 7.0
        # Quadratic ease-in-out: smooth acceleration and deceleration
        if t < 0.5:
            alpha = 2.0 * t ** 2
        else:
            alpha = 1.0 - 2.0 * (1.0 - t) ** 2
        interp_frame = (1.0 - alpha) * img_ref + alpha * img_curr
        frames_list.append(interp_frame)
    frames_list.append(img_curr)
    return frames_list

# =============================================================================


# Path configuration

pips_source_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "pips")

# Add both pips directory and its parent to sys.path to ensure proper module resolution
draggan_root = os.path.dirname(os.path.dirname(__file__))  # DragGAN root
if draggan_root not in sys.path:
    sys.path.insert(0, draggan_root)
if pips_source_path not in sys.path:
    sys.path.insert(0, pips_source_path)

try:
    # Import with proper module path
    spec = importlib.util.spec_from_file_location("pips_module", os.path.join(pips_source_path, "nets", "pips.py"))
    pips_nets_module = importlib.util.module_from_spec(spec)
    sys.modules['pips_nets'] = pips_nets_module
    spec.loader.exec_module(pips_nets_module)
    pips_core_model = pips_nets_module.Pips
    
    PIPS_AVAILABLE = True
    print(f"[PIPS]  Successfully imported PIPS core model")
except ImportError as e:
    PIPS_AVAILABLE = False
    pips_core_model = None
    print(f"[PIPS]  Import failed: {str(e)} | Fallback to SimpleRAFTTracker")
except Exception as e:
    PIPS_AVAILABLE = False
    pips_core_model = None
    print(f"[PIPS]  Unexpected import error: {str(e)}")

# RAFT model import
try:
    raft_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "RAFT")
    sys.path.insert(0, raft_dir)
    from RAFT.raft_loader import load_raft_model
    from argparse import Namespace
    
    raft_core_model, raft_loaded = load_raft_model()
    RAFT_AVAILABLE = raft_loaded and raft_core_model is not None
    
    if RAFT_AVAILABLE:
        print(f"[RAFT]  Successfully imported RAFT optical flow model")
    else:
        print(f"[RAFT]  Failed to load RAFT model")
        raft_core_model = None
        
except ImportError as e:
    RAFT_AVAILABLE = False
    raft_core_model = None
    print(f"[RAFT]  Import failed: {str(e)}")
except Exception as e:
    RAFT_AVAILABLE = False
    raft_core_model = None
    print(f"[RAFT]  Unexpected import error: {str(e)}")

# -----------------------------------------------------------------------------
# SimpleRAFTTracker
# -----------------------------------------------------------------------------
class SimpleRAFTTracker:
    def __init__(self, device, feature_scale: float = 0.125, num_iterations: int = 4):
        self.device = device
        self.feature_scale = feature_scale
        self.num_iterations = num_iterations  # keep original param, no use

    def track_points(
        self,
        feat_ref: torch.Tensor,
        feat_curr: torch.Tensor,
        points: List[List[float]],
        search_radius: int = 12,
        max_step: float = 8.0,
        spatial_sigma_scale: float = 0.5,
        min_move_threshold: float = 0.8,  # move threshold
    ) -> List[List[float]]:
        tracked_points: List[List[float]] = []
        _, C, H, W = feat_ref.shape

        for point in points:
            py, px = float(point[0]), float(point[1])
            iy, ix = round(py), round(px)

            if iy < 0 or ix < 0 or iy >= H or ix >= W:
                tracked_points.append(point)
                continue

            ref_feat = feat_ref[0, :, iy, ix]
            ref_feat_norm = ref_feat / (ref_feat.norm(dim=-1) + 1e-8)

            r = max(3, int(search_radius))
            up = max(0, iy - r)
            down = min(H, iy + r + 1)
            left = max(0, ix - r)
            right = min(W, ix + r + 1)

            if down <= up or right <= left:
                tracked_points.append(point)
                continue

            feat_patch = feat_curr[:, :, up:down, left:right]
            feat_patch_norm = feat_patch / (feat_patch.norm(dim=1, keepdim=True) + 1e-8)

            ref_feat_expanded = ref_feat_norm.view(1, C, 1, 1)
            cos_sim = (feat_patch_norm * ref_feat_expanded).sum(dim=1)

            y_coords = torch.arange(feat_patch.shape[2], device=self.device, dtype=torch.float32)
            x_coords = torch.arange(feat_patch.shape[3], device=self.device, dtype=torch.float32)
            yy, xx = torch.meshgrid(y_coords, x_coords, indexing='ij')

            # Use real relative position of tracking point in patch (not patch geometric center)
            center_y = iy - up
            center_x = ix - left
            dist_squared = (yy - center_y) ** 2 + (xx - center_x) ** 2

            # Use sqrt of patch area for sigma, spatial weight works correctly now
            patch_size = np.sqrt(feat_patch.shape[2] * feat_patch.shape[3])
            sigma = spatial_sigma_scale * patch_size
            spatial_weight = torch.exp(-dist_squared / (2.0 * sigma ** 2 + 1e-8))

            match_score = cos_sim[0] * spatial_weight

            max_score, max_flat_idx = torch.max(match_score.view(-1), dim=0)
            # Only update if match is valid (cos_sim > 0) and quality is acceptable
            valid_match = max_score > 0.0
            
            best_dy = max_flat_idx // feat_patch.shape[3]
            best_dx = max_flat_idx % feat_patch.shape[3]

            best_y = best_dy + up
            best_x = best_dx + left

            step_y = float(best_y) - py
            step_x = float(best_x) - px
            step_norm = np.sqrt(step_y ** 2 + step_x ** 2)

            if not valid_match or step_norm < min_move_threshold:
                step_y = 0.0
                step_x = 0.0
        
            elif step_norm > max_step and step_norm > 1e-6:
                step_scale = max_step / step_norm
                step_y *= step_scale
                step_x *= step_scale
            
            new_y = np.clip(py + step_y, 0.0, float(H - 1))
            new_x = np.clip(px + step_x, 0.0, float(W - 1))
            tracked_points.append([new_y, new_x])

        return tracked_points

# -----------------------------------------------------------------------------
# PIPsTracker (Added Nearest/SimpleRAFT robust logic)
# -----------------------------------------------------------------------------
class PIPsTracker:
    def __init__(self, device, model_path: Optional[str] = None, interpolation_method: str = 'cubic_spline'):
        self.device = device
        self.model = None
        self.prev_points = None
        self.interpolation_method = interpolation_method  # 'cubic_spline', 'ease_in_out', 'quadratic'
        # Initialize SimpleRAFTTracker for fallback/robust check
        self.raft_tracker = SimpleRAFTTracker(device)

        if PIPS_AVAILABLE and pips_core_model is not None:
            try:
                # Initialize PIPS with ORIGINAL structure (compatible with pre-trained weights)
                self.model = pips_core_model(stride=4)
                print("[PIPS]  PIPS model initialized with original structure (weight-compatible)")

                # Load pre-trained weights (keep original logic)
                weight_loaded = False
                target_weight_name = "model-000200000.pth"
                root_dir = os.path.dirname(os.path.dirname(__file__))  # DragGAN root
                pips_dir = os.path.join(root_dir, "pips")
                weight_paths = [
                    model_path,
                    os.path.join(pips_dir, "checkpoint", target_weight_name),
                    os.path.join(root_dir, "checkpoint", target_weight_name),
                    os.path.join(root_dir, "viz", "checkpoint", target_weight_name)
                ]

                valid_ckpt_path = None
                for wp in weight_paths:
                    if wp is not None and os.path.exists(wp):
                        valid_ckpt_path = wp
                        break

                if valid_ckpt_path is not None:
                    checkpoint = torch.load(valid_ckpt_path, map_location=self.device, weights_only=True)
                    if 'model_state_dict' in checkpoint:
                        model_state = checkpoint['model_state_dict']
                        print("[PIPS]  Extract weights from 'model_state_dict'")
                    elif 'model' in checkpoint:
                        model_state = checkpoint['model']
                        print("[PIPS]  Extract weights from 'model'")
                    else:
                        model_state = checkpoint
                        print("[PIPS]  Extract weights from root of checkpoint")
                    
                    model_keys = set(self.model.state_dict().keys())
                    new_model_state = {}
                    for k, v in model_state.items():
                        clean_k = k.replace("module.", "") if k.startswith("module.") else k
                        if clean_k in model_keys:
                            new_model_state[clean_k] = v
                        else:
                            print(f"[PIPS]  Skip unused weight key: {k} (cleaned: {clean_k})")
                    
                    self.model.load_state_dict(new_model_state, strict=True)
                    weight_loaded = True
                    print(f"[PIPS]  Pre-trained weights loaded successfully: {valid_ckpt_path}")
                    print(f"[PIPS]  Loaded {len(new_model_state)}/{len(model_state)} weight keys")
                else:
                    print(f"[PIPS]  Pre-trained weight not found (will use random weights)")

                self.model = self.model.to(self.device).eval()
                print(f"[PIPS]  PIPS tracker ready | Device: {self.device} | Weight-compatible mode")

            except Exception as e:
                print(f"[PIPS]  Initialization failed: {str(e)}")
                traceback.print_exc()
                self.model = None
        else:
            print("[PIPS]  PIPS model unavailable | Fallback to SimpleRAFTTracker")

    def track_points(
        self,
        img_ref: torch.Tensor,
        img_curr: torch.Tensor,
        points: List[List[float]],
        feat_ref: Optional[torch.Tensor] = None,
        feat_curr: Optional[torch.Tensor] = None,
        # Add robust params (same as SimpleRAFT/Nearest)
        min_move_threshold: float = 0.5,  # More sensitive, allow smaller moves
        max_step: float = 8.0,
        target_points: Optional[List[List[float]]] = None,  # Optional direction validation
    ) -> List[List[float]]:
        if self.model is None:
            print("[PIPS] Fallback to SimpleRAFTTracker (PIPS unavailable)")
            if feat_ref is not None and feat_curr is not None:
                return self.raft_tracker.track_points(feat_ref, feat_curr, points)
            return points

        with torch.no_grad():
            # Step 1: Process input frames
            img_ref = img_ref.squeeze().unsqueeze(0).to(self.device)
            img_curr = img_curr.squeeze().unsqueeze(0).to(self.device)
            img_ref = torch.clamp(img_ref, 0.0, 1.0)
            img_curr = torch.clamp(img_curr, 0.0, 1.0)
            
            # Step 1a: Generate 8-frame sequence using selected interpolation method
            if self.interpolation_method == 'cubic_spline':
                frames_list = interpolate_frames_cubic_spline(img_ref, img_curr)
            elif self.interpolation_method == 'ease_in_out':
                frames_list = interpolate_frames_easing(img_ref, img_curr, easing_type='ease_in_out')
            elif self.interpolation_method == 'quadratic':
                frames_list = interpolate_frames_quadratic(img_ref, img_curr)
            else:  # fallback to cubic spline
                frames_list = interpolate_frames_cubic_spline(img_ref, img_curr)
            
            # Stack frames properly: each frame is [1, 3, H, W], stack on dim=1 to get [1, 8, 3, H, W]
            # NOTE: torch.cat concatenates channels, we need torch.stack to preserve frame dimension
            rgb_sequence = torch.stack(frames_list, dim=1).squeeze(0)  # [1, 8, 3, H, W] -> [8, 3, H, W], then add batch
            rgb_sequence = rgb_sequence.unsqueeze(0)  # [1, 8, 3, H, W]

            # Step 2: Process input points
            points_np = np.array(points, dtype=np.float32)
            points_np = np.atleast_2d(points_np)
            points_tensor = torch.tensor(points_np[:, [1, 0]], device=self.device, dtype=torch.float32).unsqueeze(0)
            try:
                # Run PIPS inference
                model_output = self.model(points_tensor, rgb_sequence, iters=3)
                coord_predictions = model_output[0]
                final_pred = coord_predictions[-1]
                
                # Use Frame 7 for full motion range
                tracked_xy = final_pred[0, 7, :, :]  # [N,2] (x,y)
                
                # Convert to DragGAN's [y,x] format with enhanced stability
                _, _, _, H_frame, W_frame = rgb_sequence.shape
                tracked_points = []
                for idx, (x_pred, y_pred) in enumerate(tracked_xy):
                    orig_y, orig_x = points[idx]
                    pred_x = x_pred.item()
                    pred_y = y_pred.item()

                    # Clamp predictions to valid range first
                    pred_x = np.clip(pred_x, 0.0, float(W_frame - 1))
                    pred_y = np.clip(pred_y, 0.0, float(H_frame - 1))

                    # Calculate step
                    step_y = pred_y - orig_y
                    step_x = pred_x - orig_x
                    step_norm = np.sqrt(step_y ** 2 + step_x ** 2)

                    # Enhanced logic for smooth convergence:
                    # 1. If movement is very small, still allow it
                    if step_norm < 0.1:
                        # Very small movement - keep it for fine convergence
                        pass  # Use step_y, step_x as-is
                    # 2. If step is below threshold, treat as potential noise
                    elif step_norm < min_move_threshold:
                        # Likely noise - reduce but don't completely discard
                        step_y *= 0.3  # Reduce noise impact
                        step_x *= 0.3
                    # 3. If step is too large, cap it smoothly
                    elif step_norm > max_step and step_norm > 1e-6:
                        step_scale = max_step / step_norm
                        step_y *= step_scale
                        step_x *= step_scale
                    
                    # Apply step
                    new_y = np.clip(orig_y + step_y, 0.0, float(H_frame - 1))
                    new_x = np.clip(orig_x + step_x, 0.0, float(W_frame - 1))
                    tracked_points.append([new_y, new_x])

                return tracked_points

            except Exception as e:
                print(f"[PIPS]  Inference error: {str(e)}")
                traceback.print_exc()
                print("[PIPS] Fallback to SimpleRAFTTracker (safe mode)")
                if feat_ref is not None and feat_curr is not None:
                    return self.raft_tracker.track_points(feat_ref, feat_curr, points)
                return points


# =============================================================================
# RAFTTracker (Optical Flow based tracking)
# =============================================================================
class RAFTTracker:
    """
    RAFT-based optical flow tracker optimized for DragGAN's low-texture & convergence stability.
    Add: Your original L2 nearest neighbor matching along RAFT flow direction (no jitter / precise)
    """
    
    def __init__(self, device, model_path: Optional[str] = None):
        self.device = device
        self.model = None
        self.raft_args = None
    
        
        if RAFT_AVAILABLE and raft_core_model is not None:
            try:
                self.raft_args = Namespace(
                    small=False,
                    mixed_precision=False,
                    alternate_corr=False,
                    dropout=0.0
                )
                self.model = raft_core_model(self.raft_args)
                print("[RAFT]  RAFT model initialized")
                
                # Weight loading logic
                weight_loaded = False
                current_dir = os.path.dirname(os.path.abspath(__file__))
                draggan_root = os.path.dirname(current_dir)
                weight_paths = [
                    model_path,
                    os.path.join(draggan_root, "RAFT", "models", "models", "raft-things.pth"),
                    os.path.join(draggan_root, "RAFT", "models", "models", "raft-sintel.pth"),
                    os.path.join(draggan_root, "RAFT", "models", "models", "raft-kitti.pth"),
                    os.path.join(draggan_root, "RAFT", "models", "models", "raft-small.pth"),
                    os.path.join(draggan_root, "RAFT", "models", "raft-things.pth"),
                    os.path.join(draggan_root, "RAFT", "models", "raft-sintel.pth"),
                    os.path.join(draggan_root, "RAFT", "models", "raft-kitti.pth"),
                    os.path.join(draggan_root, "RAFT", "models", "raft-small.pth"),
                ]
                
                valid_ckpt_path = None
                for wp in weight_paths:
                    if wp is not None and os.path.exists(os.path.abspath(wp)):
                        valid_ckpt_path = os.path.abspath(wp)
                        break
                
                if valid_ckpt_path is not None:
                    try:
                        checkpoint = torch.load(valid_ckpt_path, map_location=self.device, weights_only=False)
                        state_dict = checkpoint
                        if all(k.startswith('module.') for k in state_dict.keys()):
                            state_dict = {k.replace('module.', '', 1): v for k, v in state_dict.items()}
                        self.model.load_state_dict(state_dict)
                        weight_loaded = True
                        print(f"[RAFT]  Weights loaded: {valid_ckpt_path}")
                    except Exception as e:
                        print(f"[RAFT]  Weight load failed: {str(e)}")
                        weight_loaded = False
                else:
                    print("[RAFT]  No valid weight path found")

                self.model = self.model.to(self.device).eval()
                print(f"[RAFT]  Ready | Device: {self.device} | Weights: {weight_loaded}")
                
            except Exception as e:
                print(f"[RAFT]  Init failed: {str(e)}")
                traceback.print_exc()
                self.model = None
        else:
            print("[RAFT]  Model unavailable (import error)")
    
    def track_points(
        self,
        img_ref: torch.Tensor,
        img_curr: torch.Tensor,
        points: List[List[float]],
        iters: int = 20,
        min_move_threshold: float = 0.2,  # Lower: capture tiny valid movement
        max_step: float = 12.0,           # Larger: match fast image deformation
        use_acceleration: bool = True,
        confidence_threshold: float = 0.15,
        target_points: Optional[List[List[float]]] = None,
        feat_ref: Optional[torch.Tensor] = None,
        feat_curr: Optional[torch.Tensor] = None,
    ) -> List[List[float]]:
        """
        Track points with ultimate stability for DragGAN.
        """
        if self.model is None:
            return points
        
        with torch.no_grad():
            try:
                # Frame preprocessing
                img_ref = img_ref.squeeze().unsqueeze(0).to(self.device)
                img_curr = img_curr.squeeze().unsqueeze(0).to(self.device)
                if img_ref.max() <= 1.0:
                    img_ref = img_ref * 255.0
                    img_curr = img_curr * 255.0
                img_ref = img_ref.to(torch.uint8)
                img_curr = img_curr.to(torch.uint8)
                _, _, H, W = img_ref.shape
                
                # RAFT flow inference
                flow_list = self.model(img_ref, img_curr, iters=iters, test_mode=True)
                flow = flow_list[-1]  # [1,2,H,W] (dx, dy)
                flow_magnitude = torch.norm(flow[0], dim=0)
                flow_valid_mask = flow_magnitude > 0.1
                flow_mean = float(flow_magnitude[flow_valid_mask].mean().item()) if flow_valid_mask.sum() > 0 else 0.0

                # Texture quality map (gradient-based)
                img_curr_np = img_curr[0].float().mean(dim=0).cpu().numpy()
                dy_grad = np.abs(np.diff(img_curr_np, axis=0))
                dx_grad = np.abs(np.diff(img_curr_np, axis=1))
                texture_quality = np.pad(dy_grad, ((0,1),(0,0)), mode='edge') + np.pad(dx_grad, ((0,0),(0,1)), mode='edge')
                texture_quality = np.clip(texture_quality / texture_quality.max(), 0.0, 1.0)

                tracked_points = []
                points_np = np.array(points, dtype=np.float32)
                has_targets = target_points is not None and len(target_points) == len(points)

                for idx, point in enumerate(points_np):
                    orig_y, orig_x = float(point[0]), float(point[1])
                    y_idx = int(np.clip(orig_y, 0, H - 1))
                    x_idx = int(np.clip(orig_x, 0, W - 1))
                    
                    # Sample raw RAFT flow
                    flow_at_point = flow[0, :, y_idx, x_idx]
                    dx_raft = float(flow_at_point[0].item()) 
                    dy_raft = float(flow_at_point[1].item())  
                    step_norm_raft = np.sqrt(dx_raft ** 2 + dy_raft ** 2)

                    step_x = 0.0
                    step_y = 0.0
                
                
                    nearest_step_x = 0.0
                    nearest_step_y = 0.0
                    if feat_ref is not None and feat_curr is not None and step_norm_raft > min_move_threshold:
                        
                        r2 = 128
                        r = round(r2 / 512 * H)
                        py, px = int(orig_y), int(orig_x)
                        
                    
                        if dy_raft > 0:
                            up = max(py, 0)
                            down = min(py + r + 1, H)
                        else:
                            up = max(py - r, 0)
                            down = min(py + 1, H)

                        if dx_raft > 0:
                            left = max(px, 0)
                            right = min(px + r + 1, W)
                        else: 
                            left = max(px - r, 0)
                            right = min(px + 1, W)
                        
                        
                        feat_patch = feat_curr[:, :, up:down, left:right]
                        ref_feat = feat_ref[:, :, y_idx, x_idx].reshape(1,-1,1,1)
                        L2 = torch.linalg.norm(feat_patch - ref_feat, dim=1)
                        _, min_idx = torch.min(L2.view(1,-1), -1)
                        width = right - left
                        match_y = min_idx.item() // width + up
                        match_x = min_idx.item() % width + left
                        
                        
                        nearest_step_y = match_y - orig_y
                        nearest_step_x = match_x - orig_x

                    # Core: Target-guided movement with stability optimizations
                    if has_targets and step_norm_raft > min_move_threshold:
                        tar_y, tar_x = target_points[idx]
                        dy_target = tar_y - orig_y
                        dx_target = tar_x - orig_x
                        target_norm = np.sqrt(dy_target ** 2 + dx_target ** 2)
                        
                        if target_norm > 1e-6:
                            # Normalize direction vectors
                            dy_target_unit = dy_target / target_norm
                            dx_target_unit = dx_target / target_norm
                            dy_raft_unit = dy_raft / step_norm_raft if step_norm_raft > 0 else 0.0
                            dx_raft_unit = dx_raft / step_norm_raft if step_norm_raft > 0 else 0.0
                            
                            # Direction consistency check (cosine similarity)
                            cos_sim = dx_raft_unit * dx_target_unit + dy_raft_unit * dy_target_unit
                            
                            if cos_sim > 0.0:
                                # 1. Direction weight: weaker decay + high floor → preserve forward movement
                                dir_weight = cos_sim * 0.8 + 0.2
                                # 2. Texture weight: STRONG FLOOR (0.5) → no stuck in low-texture areas
                                tex_weight = texture_quality[y_idx, x_idx] * 0.5 + 0.5
                                # 3. Flow weight: looser upper bound → capture more valid flow
                                flow_weight = min(flow_magnitude[y_idx, x_idx].item() / (flow_mean * 0.7 + 1e-8), 1.5)
                                # 4. Global speed boost
                                total_weight = dir_weight * tex_weight * flow_weight * 1.2

                                # Exponential decay: fast when far, slow when close
                                dist_weight = np.exp(-target_norm / 60.0)  # 60 = decay rate (tuneable)
                                total_weight = total_weight * (0.7 + 0.3 * dist_weight)

                                # Apply weighted step (RAFT)
                                step_x = dx_raft * total_weight
                                step_y = dy_raft * total_weight
                                step_norm = np.sqrt(step_x ** 2 + step_y ** 2)

                                # Step cap (prevent overshoot)
                                if step_norm > max_step:
                                    step_scale = max_step / step_norm
                                    step_x *= step_scale
                                    step_y *= step_scale

                                # When close to target (<10px), pull directly to target
                                if target_norm < 10.0:
                                    snap_strength = 1.0 - (target_norm / 10.0)  # 0→1 as distance→0
                                    step_x = dx_target * snap_strength + step_x * (1 - snap_strength)
                                    step_y = dy_target * snap_strength + step_y * (1 - snap_strength)

                                # Smooth tiny steps to avoid back-and-forth jitter
                                if step_norm < 0.5:
                                    step_x *= 0.5
                                    step_y *= 0.5
                        
                       
                        if feat_ref is not None and feat_curr is not None:
                            step_x = step_x * 0.6 + nearest_step_x * 0.4
                            step_y = step_y * 0.6 + nearest_step_y * 0.4

                    # Fallback: raw flow when no targets 
                    elif step_norm_raft > min_move_threshold:
                        step_x = dx_raft
                        step_y = dy_raft
                        step_norm = np.sqrt(step_x ** 2 + step_y ** 2)
                        if step_norm > max_step:
                            step_scale = max_step / step_norm
                            step_x *= step_scale
                            step_y *= step_scale
                    
                        if feat_ref is not None and feat_curr is not None:
                            step_x = step_x * 0.6 + nearest_step_x * 0.4
                            step_y = step_y * 0.6 + nearest_step_y * 0.4

                    # Apply movement + boundary clamp
                    new_y = orig_y + step_y
                    new_x = orig_x + step_x
                    y_clipped = np.clip(new_y, 0.0, float(H - 1))
                    x_clipped = np.clip(new_x, 0.0, float(W - 1))
                    tracked_points.append([y_clipped, x_clipped])
                
                

                return tracked_points
                
            except Exception as e:
                print(f"[RAFT]  Inference error: {str(e)}")
                traceback.print_exc()
                return points
