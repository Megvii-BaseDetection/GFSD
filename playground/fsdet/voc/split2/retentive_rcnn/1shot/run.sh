pods_train --num-gpus 8
python combine_rpn.py --novel-model log/model_final.pth
mv model_redetect.pth log/
python test_net.py --double-rpn --num-gpus 8 MODEL.WEIGHTS log/model_redetect.pth
