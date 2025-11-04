import torch
import numpy as np
import glymur
import tempfile, io
import subprocess
import os
from PIL import Image
import math
import matplotlib.pyplot as plt
from einops import rearrange

def jpeg2000_patchwise_compress(images: torch.Tensor, bpp_map: torch.Tensor):
    """
    This function does not satisfy bpp constraint strictly. The actual bpp may be larger than the target bpp.
    Patch-wise JPEG2000 compression/decompression using OpenJPEG CLI (opj_compress/opj_decompress).
    The bpp_map defines bits-per-channel for each patch.
    
    Args:
        images: torch.Tensor [B, C, H, W], float in [0, 1]
        bpp_map: torch.Tensor [B, h, w], bits per *channel* for each patch

    Returns:
        torch.Tensor [B, C, H, W] - decompressed images
    """
    assert images.dim() == 4, "images must have shape [B, C, H, W]"
    assert bpp_map.dim() == 3, "bpp_map must have shape [B, h, w]"

    B, C, H, W = images.shape
    _, h, w = bpp_map.shape
    ph, pw = H // h, W // w
    outputs = torch.zeros_like(images)

    for bi in range(B):
        for yi in range(h):
            for xi in range(w):
                patch = images[bi, :, yi*ph:(yi+1)*ph, xi*pw:(xi+1)*pw]
                patch_np = (patch.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
                patch_np = np.transpose(patch_np, (1, 2, 0))  # [H, W, C]

                # Save patch as PNG (for OpenJPEG input)
                with tempfile.TemporaryDirectory() as tmpdir:
                    input_path = os.path.join(tmpdir, "patch.png")
                    output_jp2 = os.path.join(tmpdir, "patch.jp2")
                    output_dec = os.path.join(tmpdir, "patch_dec.png")
                    Image.fromarray(patch_np).save(input_path)

                    # Target bits-per-sample (per channel)
                    bpp = float(bpp_map[bi, yi, xi])
                    if bpp <= 0.0:
                        bpp = 1e-4
                    # Compute compression ratio (8 / bpp)
                    cratio = max(8.0 / bpp, 1.0) 
                    cratio = min(cratio, 10000.0)

                    # --- Encode with OpenJPEG ---
                    cmd_encode = [
                        "opj_compress",
                        "-i", input_path,
                        "-o", output_jp2,
                        "-r", str(cratio)
                    ]
                    subprocess.run(cmd_encode, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)

                    # --- Decode back ---
                    cmd_decode = [
                        "opj_decompress",
                        "-i", output_jp2,
                        "-o", output_dec
                    ]
                    subprocess.run(cmd_decode, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)

                    # Load decompressed patch
                    dec_img = np.array(Image.open(output_dec)).astype(np.float32) / 255.0
                    if dec_img.ndim == 2:
                        dec_img = dec_img[..., None]
                    dec_img = np.transpose(dec_img, (2, 0, 1))  # [C, H, W]
                    outputs[bi, :, yi*ph:(yi+1)*ph, xi*pw:(xi+1)*pw] = torch.from_numpy(dec_img)

    return outputs


        
def psnr(x, y):
    """Compute PSNR between two [0,1]-range tensors."""
    mse = torch.mean((x - y) ** 2).item()
    if mse == 0:
        return float("inf")
    
    #10 * math.log10(1.0 / mse)
    ans = 10 * np.log(1.0 / mse)/np.log(10)
    return ans  
    
if __name__ == '__main__':
    print('---'*10)
    print("Checking glymur and openjpeg version")
    print("glymur.__version__:",glymur.__version__) # used version: 0.13.6
    print("glymur.version.openjpeg_version:",glymur.version.openjpeg_version) # used version: 2.4.0
    print('---'*10)

    
    
    
    
    
    
    
    
    
    
    
    
    
    