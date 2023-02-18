export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=0

log_name=log_bert
nohup python -u  driver/Train_baselineModel.py --config_file expdata/opinion.cfg --thread 1 --use-cuda True > $log_name 2>&1 &
tail -f $log_name