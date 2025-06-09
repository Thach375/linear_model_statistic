import os
import zipfile
import json

def check_kaggle_auth():
    """Kiểm tra file cấu hình kaggle.json"""
    kaggle_dir = os.path.expanduser('~/.kaggle')
    kaggle_file = os.path.join(kaggle_dir, 'kaggle.json')

    if not os.path.exists(kaggle_file):
        print(" Không tìm thấy kaggle.json!")
        print(" Hãy tạo API token trên kaggle.com và đặt tại:")
        print(f"   {kaggle_file}")
        return False

    try:
        with open(kaggle_file, 'r') as f:
            creds = json.load(f)
        if 'username' in creds and 'key' in creds:
            return True
        else:
            print(" kaggle.json sai định dạng.")
            return False
    except Exception as e:
        print(f"Lỗi khi đọc kaggle.json: {e}")
        return False

def download_house_price_data():
    """Tải dữ liệu House Prices từ Kaggle"""
    data_dir = "data"
    zip_path = os.path.join(data_dir, "house_prices.zip")

    if not os.path.exists(os.path.join(data_dir, "train.csv")):
        if not check_kaggle_auth():
            raise Exception(" Kaggle chưa được cấu hình đúng. Không thể tải dữ liệu.")

        os.makedirs(data_dir, exist_ok=True)

        print(" Đang tải dữ liệu từ Kaggle...")
        os.system(f"kaggle competitions download -c house-prices-advanced-regression-techniques -p {data_dir}")

        print(" Đang giải nén...")
        with zipfile.ZipFile(os.path.join(data_dir, "house-prices-advanced-regression-techniques.zip"), 'r') as zip_ref:
            zip_ref.extractall(data_dir)

        print(" Xoá file zip...")
        os.remove(os.path.join(data_dir, "house-prices-advanced-regression-techniques.zip"))
        print(" Đã tải xong dữ liệu House Prices!")
    else:
        print(" Dữ liệu đã tồn tại, không cần tải lại.")

if __name__ == "__main__":
    check_kaggle_auth()
    download_house_price_data()
