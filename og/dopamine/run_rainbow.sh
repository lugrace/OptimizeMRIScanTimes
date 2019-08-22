export CUDA_VISIBLE_DEVICES=1
export PYTHON_PATH=${PYTHON_PATH}:.
date_str=$(date "+%Y%m%d_%H%M")
case $1 in
  1)
    recon=unrolled_recon
    ;;
  2)
    recon=cs_recon
    ;;
  3)
    recon=fft_recon
    ;;
  *)
    echo 'choose a recon type'
    exit
    ;;
esac
case $2 in
  1)
    reward=discriminator_reward
    ;;
  2)
    reward=L2_reward
    ;;
  3)
    reward=L1_reward
    ;;
  *)
    echo 'choose a reward type'
    exit
    ;;
esac
case $3 in
  [1-9])
    Rval=$3
    ;;
  *)
    echo 'choose a valid R value'
    exit
    ;;
esac
gamename="$recon;$reward;$Rval"
python -um dopamine.atari.train --agent_name=rainbow --base_dir=/mnt/data/grace/dopamine/rainbow${date_str}_${recon}_${reward}_R${Rval} --gin_files='dopamine/agents/rainbow/configs/rainbow.gin' --gin_bindings='Runner.game_name="'$gamename'"'
