#!/bin/bash

#####

# pass 'CFG' argument with the model you want to use

# CFG=metaclip_b32 # META CLIP
# CFG=metaclip_b16 # META CLIP
# CFG=metaclip_l14 # META CLIP

# CFG=clip_b32 # OpenAI CLIP
# CFG=clip_b16 # OpenAI CLIP
# CFG=clip_l14 # OpenAI CLIP

#####


#####
# pass 'type' argument with the type of llm prompts you want to use

# type=gpt 
# type=mixtral

#####


#####
# pass 'emb' argument with the type of text embeddings you want to use

# emb=s_temp # a photo of a ...
# emb=ds_temp # dataset specific prompts
# emb=mpvr # meta prompting vlm prompts
#####

DATA=data
TRAINER=clip_adapt
emb="$1"
type="$2"
CFG="$3"
shift 3  # Shift to remove the processed arguments

for data in "${@}";
  do
      CUDA_VISIBLE_DEVICES=1 python main.py \
          --root ${DATA} \
          --trainer ${TRAINER} \
          --dataset-config-file configs/datasets/"${data}".yaml \
          --config-file configs/trainers/adapt/${CFG}.yaml \
          --output-dir output/${TRAINER}/${CFG}/"${data}" \
          --txt_epochs 100 \
          --lr 0.001 \
          --txt_cls 2 \
          --zero_shot \
          --text_emb "${emb}" \
          --corruption \
          --type "${type}"
done
