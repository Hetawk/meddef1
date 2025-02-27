

### meddef1_
#python main.py --data ccts --arch meddef1_ --depth '{"meddef1_": [1.0]}' --task_name attack --attack_type bim --attack_eps 0.3 --save_attacks --gpu-ids 0
#
#python main.py --data chest_xray --arch meddef1_ --depth '{"meddef1_": [1.0]}' --task_name attack --attack_type bim --attack_eps 0.3 --save_attacks --gpu-ids 0
#
#python main.py --data scisic --arch meddef1_ --depth '{"meddef1_": [1.0]}' --task_name attack --attack_type bim --attack_eps 0.3 --save_attacks --gpu-ids 0
#
#python main.py --data tbcr --arch meddef1_ --depth '{"meddef1_": [1.0]}' --task_name attack --attack_type bim --attack_eps 0.3 --save_attacks --gpu-ids 0

python main.py --data rotc --arch meddef1_ --depth '{"meddef1_": [1.0]}' --task_name attack --attack_type bim --attack_eps 0.3 --save_attacks --gpu-ids 0



## resnet
python main.py --data ccts --arch resnet --depth '{"resnet": [18]}' --task_name attack --attack_type bim --attack_eps 0.3 --save_attacks --gpu-ids 0

python main.py --data chest_xray --arch resnet --depth '{"resnet": [18]}' --task_name attack --attack_type bim --attack_eps 0.3 --save_attacks --gpu-ids 0

python main.py --data scisic --arch resnet --depth '{"resnet": [18]}' --task_name attack --attack_type bim --attack_eps 0.3 --save_attacks --gpu-ids 0

python main.py --data tbcr --arch resnet --depth '{"resnet": [18]}' --task_name attack --attack_type bim --attack_eps 0.3 --save_attacks --gpu-ids 0

python main.py --data rotc --arch resnet --depth '{"resnet": [18]}' --task_name attack --attack_type bim --attack_eps 0.3 --save_attacks --gpu-ids 0


## densenet
python main.py --data ccts --arch densenet --depth '{"densenet": [121]}' --task_name attack --attack_type bim --attack_eps 0.3 --save_attacks --gpu-ids 0

python main.py --data chest_xray --arch densenet --depth '{"densenet": [121]}' --task_name attack --attack_type bim --attack_eps 0.3 --save_attacks --gpu-ids 0

python main.py --data scisic --arch densenet --depth '{"densenet": [121]}' --task_name attack --attack_type bim --attack_eps 0.3 --save_attacks --gpu-ids 0

python main.py --data tbcr --arch densenet --depth '{"densenet": [121]}' --task_name attack --attack_type bim --attack_eps 0.3 --save_attacks --gpu-ids 0

python main.py --data rotc --arch densenet --depth '{"densenet": [121]}' --task_name attack --attack_type bim --attack_eps 0.3 --save_attacks --gpu-ids 0

