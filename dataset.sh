




python main.py --data chest_xray --arch resnet --depth '{"resnet": [18]}' --task_name attack --attack_type fgsm --attack_eps 0.3 --save_attacks


python main.py --data scisic --arch resnet --depth '{"resnet": [18]}' --task_name attack --attack_type fgsm --attack_eps 0.3 --save_attacks

python main.py --data tbcr --arch resnet --depth '{"resnet": [18]}' --task_name attack --attack_type fgsm --attack_eps 0.3 --save_attacks


python main.py --data rotc --arch resnet --depth '{"resnet": [18]}' --task_name attack --attack_type fgsm --attack_eps 0.3 --save_attacks
