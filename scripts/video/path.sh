root=<root-path>
output_root=<output-path>

if [[ "$DATASET" == "egoschema" ]]; then
    SPLIT="full"
    doc_path=${root}/datasets/EgoSchema/${SPLIT}_doc_list.json
    visual_path=${root}/datasets/EgoSchema/Egochema_videos

elif [[ "$DATASET" == "lvb" ]]; then
    SPLIT="val"
    doc_path=${root}/datasets/LongVideoBench/lvb_${SPLIT}_doc_list.json
    visual_path=${root}/datasets/LongVideoBench/videos

elif [[ "$DATASET" == "videomme" ]]; then
    SPLIT="test"
    doc_path=${root}/datasets/Video-MME/${SPLIT}_doc_list.json
    visual_path=${root}/datasets/Video-MME/data

elif [[ "$DATASET" == "nextqa" ]]; then
    SPLIT="val"
    doc_path=${root}/datasets/nextqa/mc_${SPLIT}_doc_list.json
    visual_path=${root}/datasets/nextqa/NExTVideo

elif [[ "$DATASET" == "mlvu" ]]; then
    SPLIT="dev_mc"
    doc_path=${root}/datasets/MLVU/${SPLIT}_doc_list.json
    visual_path=${root}/datasets/MLVU/MLVU/video
fi

