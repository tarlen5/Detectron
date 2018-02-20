#! /bin/bash
#
# author: Timothy C. Arlen
#         tim.arlen@geniussports.com
#
# date:   9 Feb 2018
#
# Runs $DETECTRON/contrib/infer_save_output.py for the set of defined CONFIGS
# and WEIGHTS urls
#


SCRIPTNAME="$DETECTRON/contrib/infer_save_output.py"
INPUT_IMG_DIR="/home/ubuntu/data/computer-vision/internal_experiments/20180123_scene2_rot180_left_subset/"
OUTPUT_IMG_DIR="/home/ubuntu/data/computer-vision/internal_experiments/detectron-compare"
CONFIG_DIR="$DETECTRON/configs/12_2017_baselines/"
IMG_EXT=png

# Define models to run here:
CONFIGS=(e2e_keypoint_rcnn_R-50-FPN_1x.yaml \
	 e2e_keypoint_rcnn_R-50-FPN_s1x.yaml \
	 e2e_keypoint_rcnn_R-101-FPN_1x.yaml \
	 e2e_keypoint_rcnn_R-101-FPN_s1x.yaml \
	 e2e_keypoint_rcnn_X-101-64x4d-FPN_1x.yaml \
	 e2e_keypoint_rcnn_X-101-64x4d-FPN_s1x.yaml \
	 e2e_keypoint_rcnn_X-101-32x8d-FPN_1x.yaml \
	 e2e_keypoint_rcnn_X-101-32x8d-FPN_s1x.yaml)
WEIGHTS=(https://s3-us-west-2.amazonaws.com/detectron/37697547/12_2017_baselines/e2e_keypoint_rcnn_R-50-FPN_1x.yaml.08_42_54.kdzV35ao/output/train/keypoints_coco_2014_train%3Akeypoints_coco_2014_valminusminival/generalized_rcnn/model_final.pkl \
	 https://s3-us-west-2.amazonaws.com/detectron/37697714/12_2017_baselines/e2e_keypoint_rcnn_R-50-FPN_s1x.yaml.08_44_03.qrQ0ph6M/output/train/keypoints_coco_2014_train%3Akeypoints_coco_2014_valminusminival/generalized_rcnn/model_final.pkl \
	 https://s3-us-west-2.amazonaws.com/detectron/37697946/12_2017_baselines/e2e_keypoint_rcnn_R-101-FPN_1x.yaml.08_45_06.Y14KqbST/output/train/keypoints_coco_2014_train%3Akeypoints_coco_2014_valminusminival/generalized_rcnn/model_final.pkl \
	 https://s3-us-west-2.amazonaws.com/detectron/37698009/12_2017_baselines/e2e_keypoint_rcnn_R-101-FPN_s1x.yaml.08_45_57.YkrJgP6O/output/train/keypoints_coco_2014_train%3Akeypoints_coco_2014_valminusminival/generalized_rcnn/model_final.pkl \
	 https://s3-us-west-2.amazonaws.com/detectron/37732355/12_2017_baselines/e2e_keypoint_rcnn_X-101-64x4d-FPN_1x.yaml.16_56_16.yv4t4W8N/output/train/keypoints_coco_2014_train%3Akeypoints_coco_2014_valminusminival/generalized_rcnn/model_final.pkl \
	 https://s3-us-west-2.amazonaws.com/detectron/37732415/12_2017_baselines/e2e_keypoint_rcnn_X-101-64x4d-FPN_s1x.yaml.16_57_48.Spqtq3Sf/output/train/keypoints_coco_2014_train%3Akeypoints_coco_2014_valminusminival/generalized_rcnn/model_final.pkl \
	 https://s3-us-west-2.amazonaws.com/detectron/37792158/12_2017_baselines/e2e_keypoint_rcnn_X-101-32x8d-FPN_1x.yaml.16_54_16.LgZeo40k/output/train/keypoints_coco_2014_train%3Akeypoints_coco_2014_valminusminival/generalized_rcnn/model_final.pkl \
	 https://s3-us-west-2.amazonaws.com/detectron/37732318/12_2017_baselines/e2e_keypoint_rcnn_X-101-32x8d-FPN_s1x.yaml.16_55_09.Lx8H5JVu/output/train/keypoints_coco_2014_train%3Akeypoints_coco_2014_valminusminival/generalized_rcnn/model_final.pkl)


dt=$(date "+%Y%m%d_%H%M%S")
LOGFILE="infer_multiple_$dt.log"
touch $LOGFILE
echo "[INFO] Saving output to $LOGFILE"

for idx in `seq 1 ${#CONFIGS[@]}`; do
    idx=$((idx-1))

    config=${CONFIGS[idx]}
    weights=${WEIGHTS[idx]}

    # Confirm that config is contained in the weights string (basic sanity check)
    # Otherwise, warn user and quit
    if [[ "$weights" =~ "$config" ]]; then
	echo "[INFO] running model: $config..."
    else	
	echo "incorrect config or weights file?"
	echo "  config: $config"
	echo "  weights: $weights"
	break
    fi

    config_path=$CONFIG_DIR$config
    python $SCRIPTNAME $INPUT_IMG_DIR --cfg $config_path --wts $weights --output-dir $OUTPUT_IMG_DIR --image-ext $IMG_EXT >> $LOGFILE 2>&1
    
done
