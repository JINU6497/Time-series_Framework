from data_provider.data_loader import Dataset_custom
from torch.utils.data import DataLoader

data_dict = {
    'custom': Dataset_custom
}

def data_builder(args, type):
    
    Data = data_dict[args.data]
    # time_encode = 0 if args.embed != 'timeF' else 1
    time_encode = 0
    train_only = args.train_only

    if type == 'test':
        shuffle_flag = False
        drop_last = False
        batch_size = args.batch_size
        freq = args.freq
        
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size
        freq = args.freq

# Dataset 관련 arg 추가하면 여따 추가

    data_set = Data(
        data_type = args.data_type,
        root_path = args.root_path,
        data_path = args.data_path,
        type = type,
        len_info = [args.window_len, args.label_len, args.pred_len],
        features = args.features,
        target = args.target,
        time_encode = time_encode,
        freq = freq,
        train_only = train_only,
        scaler = args.scaler,
        train_ratio = args.train_ratio,
        test_ratio = args.test_ratio,
    )
    
    print(type, len(data_set))
    
    data_loader = DataLoader(
        data_set,
        batch_size = batch_size,
        shuffle = shuffle_flag,
        num_workers = args.num_workers,
        drop_last = drop_last)
    
    return data_set, data_loader

                