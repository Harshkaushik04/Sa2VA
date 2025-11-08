Guide to start finetuning on Ref-coco dataset(run these commands):

git clone [https://github.com/Harshkaushik04/Sa2VA.git](https://github.com/Harshkaushik04/Sa2VA.git)

cd Sa2VA

chmod +x req.sh

./req.sh

cd Sa2VA

bash tools/dist.sh train projects/sa2va/configs/my_finetune.py 1

-you can change number of gpus according to your need which is last parameter of last command
