root=<root-path>
output_root=<output-path>

if [[ "$DATASET" == "slidevqa" ]]; then
    # SPLIT="dev"
    SPLIT="test"
    doc_path=${root}/datasets/SlideVQA/${SPLIT}_doc.json
    visual_path=${root}/datasets/SlideVQA/images
elif [[ "$DATASET" == "mmlbdoc" ]]; then
    SPLIT="old"
    doc_path=${root}/datasets/MMLongBench-Doc/data/${SPLIT}_doc.json
    visual_path=${root}/datasets/MMLongBench-Doc/data/images
elif [[ "$DATASET" == "mpdocvqa" ]]; then
    # SPLIT="val"
    SPLIT="test"
    doc_path=${root}/datasets/MP-DocVQA/${SPLIT}_doc.json
    visual_path=${root}/datasets/MP-DocVQA/images
fi

