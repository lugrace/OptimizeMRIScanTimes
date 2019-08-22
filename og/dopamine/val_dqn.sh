export CUDA_VISIBLE_DEVICES=1
export PYTHON_PATH=${PYTHON_PATH}:.
#date_str=$(date "+%Y%m%d_%H%M")
[[ -z "$1" ]] && { echo "Parameter 1 is empty" ; exit 1; }
date_str=$1
python -um dopamine.atari.train --agent_name=dqn --base_dir=/mnt/data/grace/dopamine/recon$date_str --gin_files='dopamine/agents/dqn/configs/dqn.gin'
