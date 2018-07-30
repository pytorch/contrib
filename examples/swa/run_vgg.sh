#python3 train.py --dir=/scratch/pi49/pt_swa/vgg/c10/run1 --dataset=CIFAR10 --data_path=/scratch/datasets/ --model=VGG16 --epochs=300 --lr_init=0.05 --wd=5e-4 --swa --swa_start=161 --swa_lr=0.01
#mv /scratch/pi49/pt_swa/vgg/c10/run1 ~/from_scratch/pt_swa/vgg/c10
python3 train.py --dir=/scratch/pi49/pt_swa/vgg/c10/run2 --dataset=CIFAR10 --data_path=/scratch/datasets/ --model=VGG16 --epochs=300 --lr_init=0.05 --wd=5e-4 --swa --swa_start=161 --swa_lr=0.01
mv /scratch/pi49/pt_swa/vgg/c10/run2 ~/from_scratch/pt_swa/vgg/c10
