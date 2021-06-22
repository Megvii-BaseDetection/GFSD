coco_pretrain="\/path\/to\/your\/model.pth";
voc_split1_pretrain="\/path\/to\/your\/model.pth";
voc_split2_pretrain="\/path\/to\/your\/model.pth";
voc_split3_pretrain="\/path\/to\/your\/model.pth";

cd playground/fsdet/coco/
sed -i s/"\/path\/to\/your\/model.pth"/$coco_pretrain/g `grep -rl --include="config.py" ./`;
sed -i s/"\/path\/to\/your\/model.pth"/$coco_pretrain/g `grep -rl --include="combine_rpn.py" ./`;

cd ../voc/split1/
sed -i s/"\/path\/to\/your\/model.pth"/$voc_split1_pretrain/g `grep -rl --include="config.py" ./`;
sed -i s/"\/path\/to\/your\/model.pth"/$voc_split1_pretrain/g `grep -rl --include="combine_rpn.py" ./`;

cd ../../voc/split2/
sed -i s/"\/path\/to\/your\/model.pth"/$voc_split2_pretrain/g `grep -rl --include="config.py" ./`;
sed -i s/"\/path\/to\/your\/model.pth"/$voc_split2_pretrain/g `grep -rl --include="combine_rpn.py" ./`;

cd ../../voc/split3/
sed -i s/"\/path\/to\/your\/model.pth"/$voc_split3_pretrain/g `grep -rl --include="config.py" ./`;
sed -i s/"\/path\/to\/your\/model.pth"/$voc_split3_pretrain/g `grep -rl --include="combine_rpn.py" ./`;
