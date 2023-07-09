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
model_name=( DLinear )

window_len=100
pred_len=1

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
    --features MS \
    --target OT \
    --window_len $window_len \
    --label_len 0 \
    --pred_len $pred_len \
    --enc_in 9 \
    --scaler minmax \
    --des 'Exp' \
    --RevIN True \
    --num_workers 0 \
    --itr 1 --train_epochs 30 --batch_size 64 --learning_rate 0.0001
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
    --features MS \
    --target OT \
    --window_len $window_len \
    --label_len 0 \
    --pred_len $pred_len \
    --enc_in 9 \
    --des 'Exp' \
    --RevIN True \
    --num_workers 0 \
    --itr 1 --train_epochs 30 --batch_size 64 --learning_rate 0.0001
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
    --features MS \
    --target OT \
    --window_len $window_len \
    --label_len 0 \
    --pred_len $pred_len \
    --enc_in 9 \
    --scaler minmax \
    --des 'Exp' \
    --num_workers 0 \
    --itr 1 --train_epochs 30 --batch_size 64 --learning_rate 0.0001
done

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
    --features MS \
    --target OT \
    --window_len $window_len \
    --label_len 0 \
    --pred_len $pred_len \
    --enc_in 9 \
    --des 'Exp' \
    --num_workers 0 \
    --itr 1 --train_epochs 30 --batch_size 64 --learning_rate 0.0001
done