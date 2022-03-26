machine="t3"
note="cosine"
epoch=19
while [ $epoch -le 36 ]
do
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 ./tools/dist_test.sh ./configs/swin/swin-b_ms-test_max-1000.py "./work_dirs/${note}-cascade_rcnn_swin-b-p4-w7_fpn_fp16_ms-crop-3x_coco_dota/epoch_${epoch}.pth" 8 --out "/home/marina/Workspace/DOTA_devkit-master/${machine}-${note}-${epoch}.pkl" --eval bbox
    let epoch++
done
