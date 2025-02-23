




python main.py --data chest_xray --arch densenet --depth '{"densenet": [121]}' --task_name attack --attack_type fgsm --attack_eps 0.3 --save_attacks


python main.py --data scisic --arch densenet --depth '{"densenet": [121]}' --task_name attack --attack_type fgsm --attack_eps 0.3 --save_attacks

python main.py --data tbcr --arch densenet --depth '{"densenet": [121]}' --task_name attack --attack_type fgsm --attack_eps 0.3 --save_attacks


python main.py --data rotc --arch densenet --depth '{"densenet": [121]}' --task_name attack --attack_type fgsm --attack_eps 0.3 --save_attacks
