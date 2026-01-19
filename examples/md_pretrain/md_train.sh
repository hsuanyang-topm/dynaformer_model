# set -o xtrace
# set -x

ulimit -c unlimited
ROOT_DIR=$(realpath "$(dirname "$0")/../..")
if [ -f "/root/miniconda3/etc/profile.d/conda.sh" ]; then
  . /root/miniconda3/etc/profile.d/conda.sh
  conda activate dynaformer_py39
  export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
  export LD_LIBRARY_PATH=$CONDA_PREFIX/lib/python3.9/site-packages/torch/lib:$LD_LIBRARY_PATH
fi
if [ -z "${n_gpu}" ]; then
  if [ -n "${CUDA_VISIBLE_DEVICES}" ]; then
    n_gpu=$(echo "${CUDA_VISIBLE_DEVICES}" | awk -F',' '{print NF}')
  else
    n_gpu=$(python - <<'PY'
import torch
print(torch.cuda.device_count())
PY
)
    if [ -z "$n_gpu" ] || [ "$n_gpu" -eq 0 ]; then
      n_gpu=$(nvidia-smi -L | wc -l)
    fi
    if [ "$n_gpu" -gt 0 ]; then
      CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((n_gpu - 1)))
      export CUDA_VISIBLE_DEVICES
    fi
  fi
fi
[ -z "${lr}" ] && lr=1e-4
[ -z "${end_lr}" ] && end_lr=1e-9
[ -z "${max_epoch}" ] && max_epoch=100
[ -z "${layers}" ] && layers=4
[ -z "${hidden_size}" ] && hidden_size=512
[ -z "${ffn_size}" ] && ffn_size=512
[ -z "${num_head}" ] && num_head=32
[ -z "${batch_size}" ] && batch_size=3
[ -z "${clip_norm}" ] && clip_norm=5
[ -z "${num_workers}" ] && num_workers=16

[ -z "${update_freq}" ] && update_freq=1
[ -z "${total_steps}" ] && total_steps=$((20000*(max_epoch+1)/batch_size/n_gpu/update_freq))
[ -z "${warmup_steps}" ] && warmup_steps=$((total_steps*10/100))
[ -z "${seed}" ] && seed=2022

[ -z "${dataset_name}" ] && dataset_name="mddata:set_name=md-refined2026-5-5-5,seed=2022"
if [ -z "${data_path}" ]; then
  if [ -d "$ROOT_DIR/Dataset" ]; then
    data_path=$(realpath "$ROOT_DIR/Dataset")
  elif [ -d "/home/hyang/Dataset/output" ]; then
    data_path="/home/hyang/Dataset/output"
  # elif [ -d "/root/dataset" ]; then
  #   data_path="/root/dataset"
  elif [ -d "/home/hyang/Dynaformer/output" ]; then
    data_path="/home/hyang/Dynaformer/output"
  else
    data_path=""
  fi
fi
if [ -z "${require_processed}" ]; then
  require_processed=true
fi
# if [ "$require_processed" = "true" ] && [ -n "$data_path" ]; then
#   if [[ "$dataset_name" == mddata:* ]]; then
#     set_name=$(echo "$dataset_name" | sed -n 's/.*set_name=\([^,]*\).*/\1/p')
#     if [ -z "$set_name" ]; then
#       echo "ERROR: cannot parse set_name from dataset_name=$dataset_name"
#       exit 1
#     fi
#     train_pkl="${data_path}/processed_${set_name}_train.pkl"
#     valid_pkl="${data_path}/processed_${set_name}_valid.pkl"
#     test_pkl="${data_path}/processed_${set_name}_test.pkl"
#     missing=0
#     for f in "$train_pkl" "$valid_pkl" "$test_pkl"; do
#       if [ ! -f "$f" ]; then
#         echo "ERROR: missing processed file: $f"
#         missing=1
#       fi
#     done
#     if [ "$missing" -ne 0 ]; then
#       echo "ERROR: processed mddata files are missing; refusing to auto-process."
#       exit 1
#     fi
#   fi
# fi
if [ -z "${save_path}" ]; then
  if [ -d "$ROOT_DIR/Train" ]; then
    save_path=$(realpath "$ROOT_DIR/Train")
  else
    save_path="/root"
  fi
fi
[ -z "${dropout}" ] && dropout=0.1
[ -z "${act_dropout}" ] && act_dropout=0.1
[ -z "${attn_dropout}" ] && attn_dropout=0.1
[ -z "${weight_decay}" ] && weight_decay=0.00001
[ -z "${sandwich_ln}" ] && sandwich_ln="false"

[ -z "${adam_betas}" ] && adam_betas="(0.9,0.999)"
[ -z "${adam_eps}" ] && adam_eps=1e-8

[ -z "${save_prefix}" ] && save_prefix="MD"
[ -z "${flag}" ] && flag=true
[ -z "${flag_m}" ] && flag_m=3
[ -z "${flag_step_size}" ] && flag_step_size=0.001
[ -z "${flag_mag}" ] && flag_mag=0.01

[ -z "${dist_head}" ] && dist_head="gbf3d"
[ -z "${num_dist_head_kernel}" ] && num_dist_head_kernel=256
[ -z "${num_edge_types}" ] && num_edge_types=$((512*32))
[ -z "${max_nodes}" ] && max_nodes=1500
[ -z "${task}" ] && task="graph_prediction"
[ -z "${loss}" ] && loss="l2_loss"
[ -z "${patience}" ] && patience="50"

[ -z "${fingerprint}" ] && fingerprint="true"

[ -z "${test_set}" ] && test_set="mddata:set_name=md-refined2026-5-5-5,seed=2022"
[ -z "${ddp_options}" ] && ddp_options=""


if [ "$flag" = 'true' ]; then
  task="${task}_with_flag"
  loss="${loss}_with_flag"
fi

OLDIFS=$IFS
IFS=':' #setting comma as delimiter
read -a strarr <<< "$dataset_name" #reading str as an array as tokens separated by IFS
IFS=$OLDIFS

hyperparams="${save_prefix}-dataset${strarr[0]}_${strarr[1]}-lr$lr-end_lr$end_lr-epoch${max_epoch}-task${task}-loss${loss}"
hyperparams+="-L${layers}D${hidden_size}F${ffn_size}H${num_head}BS$((batch_size*n_gpu*update_freq))CLIP${clip_norm}"
hyperparams+="/dp${dropout}-attn_dp${attn_dropout}-act_dp${act_dropout}-wd${weight_decay}-sandwich${sandwich_ln}"
if [ "$flag" = 'true' ]; then
  hyperparams+="-flag${flag}-flag_m${flag_m}-stepsize${flag_step_size}-mag${flag_mag}"
fi

hyperparams+="-dist_head${dist_head}"
if [ "${dist_head}" = "gbf" ] || [ "${dist_head}" = "gbf3d" ]; then
  hyperparams+="-ndhk${num_dist_head_kernel}-net${num_edge_types}"
fi
hyperparams+="-SEED${seed}"


save_dir=$save_path/$hyperparams
tsb_dir=$save_dir/tsb
[ -z "${WANDB_PROJECT_NAME}" ] && WANDB_PROJECT_NAME="MD-bind"
export WANDB_NAME="$hyperparams"
export WANDB_RUN_ID=`echo -n $save_dir | md5sum | awk '{print $1}'`
export WANDB_SAVE_DIR=$save_dir/wandb

mkdir -p "$save_dir"
mkdir -p "$WANDB_SAVE_DIR"

echo -e "\n\n"
echo "=====================================ARGS======================================"
echo "arg0: $0"
echo "seed: ${seed}"
echo "batch_size: $((batch_size*n_gpu*update_freq))"
echo "num_workers: ${num_workers}"
echo "n_layers: ${layers}"
echo "lr: ${lr}"
echo "warmup_steps: ${warmup_steps}"
echo "total_steps: ${total_steps}"
echo "max_epoch: ${max_epoch}"
echo "clip_norm: ${clip_norm}"
echo "hidden_size: ${hidden_size}"
echo "ffn_size: ${ffn_size}"
echo "num_head: ${num_head}"
echo "update_freq: ${update_freq}"
echo "dropout: ${dropout}"
echo "attn_dropout: ${attn_dropout}"
echo "act_dropout: ${act_dropout}"
echo "weight_decay: ${weight_decay}"
echo "adam_betas: ${adam_betas}"
echo "adam_eps: ${adam_eps}"
echo "flag_m: ${flag_m}"
echo "flag_step_size: ${flag_step_size}"
echo "flag_mag: ${flag_mag}"
echo "save_dir: ${save_dir}"
echo "tsb_dir: ${tsb_dir}"
echo "data_dir: ${data_path}"
echo "dist_head: ${dist_head}"
echo "num_dist_head_kernel: $num_dist_head_kernel"
echo "num_edge_types: $num_edge_types"
echo "max_nodes: $max_nodes"
echo "==============================================================================="

# ENV
echo -e "\n\n"
echo "======================================ENV======================================"
echo 'Environment'
ulimit -c unlimited;
echo -e "\n\nhostname"
hostname
echo -e "\n\nnvidia-smi"
nvidia-smi
echo "which python"
which python
echo "PATH"
echo $PATH
echo "torch version"
python -c "import torch; print(torch.__version__)"
echo "ddp_options"
echo $ddp_options
echo "==============================================================================="

echo -e "\n\n"
echo "==================================ACTION ARGS==========================================="
action_args=""
if [ "$sandwich_ln" = "true" ]; then
  action_args+="--sandwich-ln "
fi

if [ "$flag" = 'true' ]; then
  action_args+="--flag-m $flag_m --flag-step-size $flag_step_size --flag-mag $flag_mag "
fi

if [ "$fingerprint" = 'true' ]; then
  action_args+="--fingerprint "
fi

echo "action_args: ${action_args}"
echo "========================================================================================"


export PYTHONPATH="$ROOT_DIR/fairseq:$PYTHONPATH"

torchrun --nproc_per_node=${n_gpu} --master_port 29501 ${ddp_options} \
  -m fairseq_cli.train \
  --user-dir "$ROOT_DIR/dynaformer" \
  --num-workers ${num_workers} --ddp-backend=legacy_ddp \
  --dataset-name "$dataset_name" \
  --dataset-source pyg --data-path "$data_path" \
  --batch-size $batch_size --data-buffer-size 50 \
  --task $task --criterion $loss --arch graphormer_base --num-classes 1 \
  --lr $lr --end-learning-rate $end_lr --lr-scheduler polynomial_decay --power 1 \
  --warmup-updates $warmup_steps --total-num-update $total_steps --max-update $total_steps --update-freq $update_freq --patience $patience \
  --encoder-layers $layers --encoder-attention-heads $num_head \
  --encoder-embed-dim $hidden_size --encoder-ffn-embed-dim $ffn_size \
  --attention-dropout $attn_dropout --act-dropout $act_dropout --dropout $dropout --weight-decay $weight_decay \
  --optimizer adam --adam-betas $adam_betas --adam-eps $adam_eps $action_args --clip-norm $clip_norm \
  --fp16 --save-dir "$save_dir" --tensorboard-logdir $tsb_dir --seed $seed \
  --max-nodes $max_nodes --dist-head $dist_head \
  --num-dist-head-kernel $num_dist_head_kernel --num-edge-types $num_edge_types 2>&1 | tee "$save_dir/train_log.txt"
