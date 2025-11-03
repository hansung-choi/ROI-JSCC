# ROI-JSCC

Implementations of main experiments for the paper "Region-of-Interest-Guided Deep Joint Source-Channel Coding for Image Transmission"

Experiment baselines are mainly based on the SOTA deepJSCC paper "Feature Importance-Aware Deep Joint Source-Channel Coding for Computationally Efficient and Adjustable Image Transmission".

## Requirements
1. python 3.8.8
2. pytorch 2.0.1
3. cuda 11.1.1
4. numpy 1.24.4
5. hydra 1.1

## Experiment code manual

### Arguments for terminal execution
1. **chan_type**: The type of communication channel, which can be **"AWGN","Rayleigh"**.
2. **rcpp**: The reciprocal of **cpp** (channel usage per RGB pixels). It can take one of the following discrete values: **12, 16, 24, or 32**.
3. **SNR_info**: The channel SNR value, which can be one of **1, 4, 7, or 10** dB.
4. **performance_metric**: The performance metric to be maximized, which can be **"PSNR","SSIM"**.
5. **data_info**: The dataset name (possible value: **"DIV2K"**).
6. **model_name**: The model name, which can be one of the following: **"ConvJSCC", "ResJSCC", "SwinJSCC", "FAJSCC", "ROIJSCC", "ROIJSCCwoRB","ROIJSCCall", "ROIJSCCnone","FAJSCCwRLB","FAJSCCwRB", or "FAJSCCwRL"**.


### Example of training a model.

    python3 main_train.py rcpp=12 chan_type=AWGN performance_metric=PSNR SNR_info=4 model_name=ConvJSCC data_info=DIV2K


### Example of experimental results for "PSNR (ROI) and PSNR (Avg) Results".
**You can obtain test results for other settings by simply modifying the rcpp values in the main_total_eval_OAset1.py file**

    python3 main_total_eval_OAset.py chan_type="AWGN" performance_metric="PSNR" data_info=DIV2K

### Example of experimental results for "PSNR (ROI) Results of the ablation study".
**You can obtain test results for other settings by simply modifying the rcpp values in the main_total_eval_Abset1.py file**

    python3 main_total_eval_Abset.py chan_type="AWGN" performance_metric="PSNR" data_info=DIV2K


### Example of experimental results for "Visual Inspection".

    python3 main_model_visualize.py  SNR_info=1 rcpp=12 chan_type="AWGN" performance_metric="PSNR" model_name=ResJSCC data_info=DIV2K



