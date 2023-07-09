export CUDA_VISIBLE_DEVICES=0

if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi

root_path_name=./dataset/Hackathon/
data_path_name=ko_KACO1.csv
model_id_name=ko_KACO1
data_name=custom
model_name=( TimesNet )

window_len=100
pred_len=1

# scaler x + revin x
for model in ${model_name[@]}
do
    python main.py \
    --is_training 1 \
    --data_type 'Forecasting' \
    --root_path $root_path_name \
    --data_path $data_path_name \
    --model_id $model_id_name_$window_len'_'$pred_len'_ko' \
    --model $model_name \
    --data $data_name \
    --features M \
    --window_len $window_len \
    --label_len 0 \
    --pred_len $pred_len \
    --enc_in 9 \
    --c_out 9 \
    --d_model 64 \
    --d_ff 64 \
    --top_k 5 \
    --num_kernels 6 \
    --des 'Exp' \
    --num_worker 0 \
    --itr 1 --train_epochs 10 --batch_size 64 --learning_rate 0.01
done

# scaler o + revin x
for model in ${model_name[@]}
do
    python main.py \
    --is_training 1 \
    --data_type 'Forecasting' \
    --root_path $root_path_name \
    --data_path $data_path_name \
    --model_id $model_id_name_$window_len'_'$pred_len'_ko_scaler' \
    --model $model_name \
    --data $data_name \
    --features M \
    --window_len $window_len \
    --label_len 0 \
    --pred_len $pred_len \
    --enc_in 9 \
    --c_out 9 \
    --d_model 64 \
    --d_ff 64 \
    --top_k 5 \
    --scaler minmax \
    --num_kernels 6 \
    --des 'Exp' \
    --num_worker 0 \
    --itr 1 --train_epochs 10 --batch_size 64 --learning_rate 0.01
done


# scaler x + revin o
for model in ${model_name[@]}
do
    python main.py \
    --is_training 1 \
    --data_type 'Forecasting' \
    --root_path $root_path_name \
    --data_path $data_path_name \
    --model_id $model_id_name_$window_len'_'$pred_len'_ko_revin' \
    --model $model_name \
    --data $data_name \
    --features M \
    --window_len $window_len \
    --label_len 0 \
    --pred_len $pred_len \
    --enc_in 9 \
    --c_out 9 \
    --d_model 64 \
    --d_ff 64 \
    --top_k 5 \
    --num_kernels 6 \
    --des 'Exp' \
    --RevIN True \
    --num_worker 0 \
    --itr 1 --train_epochs 10 --batch_size 64 --learning_rate 0.01
done

# scaler o + revin o
for model in ${model_name[@]}
do
    python main.py \
    --is_training 1 \
    --data_type 'Forecasting' \
    --root_path $root_path_name \
    --data_path $data_path_name \
    --model_id $model_id_name_$window_len'_'$pred_len'_ko_scaler+revin' \
    --model $model_name \
    --data $data_name \
    --features M \
    --window_len $window_len \
    --label_len 0 \
    --pred_len $pred_len \
    --enc_in 9 \
    --c_out 9 \
    --d_model 64 \
    --d_ff 64 \
    --top_k 5 \
    --scaler minmax \
    --num_kernels 6 \
    --des 'Exp' \
    --RevIN True \
    --num_worker 0 \
    --itr 1 --train_epochs 10 --batch_size 64 --learning_rate 0.01
done