from scipy.io import savemat, loadmat
import os
import tarfile
import pickle
import numpy as np

#%%
def save_mat(folder_name, file_name, pop, stat, W, RP):
    os.makedirs(folder_name, exist_ok=True)
    savemat(os.path.join(folder_name, file_name), {
        'pop': pop,
        'stat': stat,
        'W': W,
        'RP': RP
    })
    
#%%
def load_mat(folder_name, file_name):
    # Đảm bảo đường dẫn file tồn tại
    file_path = os.path.join(folder_name, file_name)
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File {file_path} không tồn tại.")
    
    # Tải dữ liệu từ file .mat với cấu trúc đầy đủ
    data = loadmat(file_path, struct_as_record=False, squeeze_me=True)
    
    # Trích xuất các biến cần thiết
    pop = data['pop']
    pop_dicts = [matlab_struct_to_dict(item) for item in pop]
    stat = data['stat']
    W = data['W']
    # RP = data['RP']
    
    # Trả về các biến dưới dạng dictionary
    return {
        'pop': pop_dicts,
        'stat': stat,
        'W': W,
        # 'RP': RP
    }

# %% chuyển từ struct kiểu MATLAB qua dict kiểu Python
def matlab_struct_to_dict(struct):
    return {field: getattr(struct, field) for field in struct._fieldnames}

# %% load spyder data
def load_spydata(filename):
    """
    Load Spyder .spydata saved as TAR archive with pax headers.
    """
    with tarfile.open(filename, "r") as tar:
        members = tar.getmembers()

        # tìm file pickle bên trong
        for m in members:
            if m.name.endswith(".pickle") or m.name.endswith(".pkl"):
                f = tar.extractfile(m)
                obj = pickle.load(f)
                if isinstance(obj, dict) and "globals" in obj:
                    return obj["globals"]
                return restore_globals(obj)

        raise ValueError("Không tìm thấy file pickle trong spydata.")
        
        
def restore_spyder_array(obj):
    if not isinstance(obj, dict):
        return obj

    # Spyder save-array format
    if '__save_array' in obj or '__restore_array__' in obj:
        shape = obj.get('__shape__') or obj.get('shape')
        dtype = obj.get('__dtype__') or obj.get('dtype')
        data = obj.get('__data__') or obj.get('data')

        return np.frombuffer(data, dtype=np.dtype(dtype)).reshape(shape)

    return obj

def restore_globals(globals_dict):
    restored = {}
    for k, v in globals_dict.items():
        restored[k] = restore_spyder_array(v)
    return restored
