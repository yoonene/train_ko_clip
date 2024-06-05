from datasets import load_dataset

ds = load_dataset(
    "yoonene/AI_Hub_Ko-En_Parallel_Corpus",
    split="train+valid",
    # use_auth_token=True,
    # private=True,
)

print(ds)
for idx in range(3):
    ko: str = ds[idx]["ko"]
    en: str = ds[idx]["en"]
    print(ko)
    print(en)
    print("-" * 50)
