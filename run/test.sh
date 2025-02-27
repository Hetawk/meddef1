

## MedDef 1.0
### Normal Training - Normal image
python test.py --data rotc --arch meddef1_ --depth 1.0 --model_path "out/normal_training/rotc/meddef1__1.0/save_model/best_meddef1__1.0_rotc_epochs100_lr0.001_batch64_20250226.pth" --image_path "out/normal_training/rotc/meddef1__1.0/attack/fgsm/sample_1_orig.png"

### Normal Training adv image
python test.py --data rotc --arch meddef1_ --depth 1.0 --model_path "out/normal_training/rotc/meddef1__1.0/save_model/best_meddef1__1.0_rotc_epochs100_lr0.001_batch64_20250226.pth" --image_path "out/normal_training/rotc/meddef1__1.0/attack/fgsm/sample_1_adv.png"


### Adversarial Training normal image
#### ROTC
python test.py --data rotc --arch meddef1_ --depth 1.0 --model_path "out/normal_training/rotc/meddef1__1.0/adv/save_model/best_meddef1__1.0_rotc_epochs100_lr0.0001_batch256_20250225.pth" --image_path "out/normal_training/rotc/meddef1__1.0/attack/fgsm/sample_1_orig.png"

#### ChestXray
python test.py --data chest_xray --arch meddef1_ --depth 1.0 --model_path "out/normal_training/chest_xray/meddef1__1.0/save_model/best_meddef1__1.0_chest_xray_epochs100_lr0.0003_batch16_20250227.pth" --image_path "out/normal_training/chest_xray/meddef1__1.0/attack/fgsm/sample_0_orig.png"

### Adversarial Training adv image
python test.py --data rotc --arch meddef1_ --depth 1.0 --model_path "out/normal_training/rotc/meddef1__1.0/adv/save_model/best_meddef1__1.0_rotc_epochs100_lr0.0001_batch256_20250225.pth" --image_path "out/normal_training/rotc/meddef1__1.0/attack/fgsm/sample_1_adv.png"

