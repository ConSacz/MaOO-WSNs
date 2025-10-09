from scipy.io import savemat, loadmat
import os

#%%
def save_mat(folder_name, file_name, pop, stat, MaxIt):
    os.makedirs(folder_name, exist_ok=True)
    savemat(os.path.join(folder_name, file_name), {
        'pop': pop,
        'stat': stat,
        'MaxIt': MaxIt
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
    MaxIt = data['MaxIt']
    
    # Trả về các biến dưới dạng dictionary
    return {
        'pop': pop_dicts,
        'stat': stat,
        'MaxIt': MaxIt
    }

# %% chuyển từ struct kiểu MATLAB qua dict kiểu Python
def matlab_struct_to_dict(struct):
    return {field: getattr(struct, field) for field in struct._fieldnames}

