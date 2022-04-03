machine="t1"
note="save_1"
epoch=7
while [ $epoch -le 12 ]
do
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 ./tools/dist_test.sh ./configs/swin/swin-b_ms-test_max-1000.py "./work_dirs/${note}/epoch_${epoch}.pth" 8 --out "/home/marina/Workspace/DOTA_devkit-master/${machine}_${note}_${epoch}.pkl" --eval bbox
    let epoch++
done
