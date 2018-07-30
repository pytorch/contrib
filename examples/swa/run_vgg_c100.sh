python3 train.py --dir=/scratch/pi49/pt_swa/vgg/c100/run4 --dataset=CIFAR100 --data_path=/scratch/datasets/ --model=VGG16 --epochs=300 --lr_init=0.05 --wd=5e-4 --swa --swa_start=161 --swa_lr=0.01
mv /scratch/pi49/pt_swa/vgg/c100/run4 ~/from_scratch/pt_swa/vgg/c100
python3 train.py --dir=/scratch/pi49/pt_swa/vgg/c100/run5 --dataset=CIFAR100 --data_path=/scratch/datasets/ --model=VGG16 --epochs=300 --lr_init=0.05 --wd=5e-4 --swa --swa_start=161 --swa_lr=0.01
mv /scratch/pi49/pt_swa/vgg/c100/run5 ~/from_scratch/pt_swa/vgg/c100
