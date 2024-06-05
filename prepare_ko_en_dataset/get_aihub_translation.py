from glob import glob
import pandas as pd
from datasets import load_dataset, Dataset, load_from_disk, DatasetDict
from tqdm import tqdm


ko_en_data_path = "ko_en_parallel"
file_list = glob(ko_en_data_path + "/*.xlsx")
print(len(file_list))

train_list, valid_list = [], []

for i, file in tqdm(enumerate(file_list)):
    print(file)
    data = pd.read_excel(file)
    for _, row in data.iterrows():
        ko = row["원문"].strip()
        en = row["번역문"].strip()
        if i >= (len(file_list) - 1):
            valid_list.append({"ko": ko, "en": en})
        else:
            train_list.append({"ko": ko, "en": en})
    # break

# dataset_dict = {"column_names": ["ko", "en"], "data": data_list}

train_dataset = Dataset.from_list(train_list)
valid_dataset = Dataset.from_list(valid_list)
print("TRAIN DATASET")
print(train_dataset)
print("-" * 50)
print("VALID DATASET")
print(valid_dataset)
print("-" * 50)

dataset_dict = DatasetDict({"train": train_dataset, "valid": valid_dataset})
print("DATASET_DICT")
print(dataset_dict)

dataset_dict.save_to_disk("./ko_en_parallel_dataset")
# dataset = load_from_disk("./ko_en_parallel_dataset")  # 로컬에 저장된 데이터셋 불러오기

dataset_dict.push_to_hub(
    "yoonene/AI_Hub_Ko-En_Parallel_Corpus",
    private=True,
    # use_auth_token="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",
)
print("저장 완.")
