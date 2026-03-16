from datasets import load_from_disk

dataset = load_from_disk(r"C:\Users\my tech\Documents\3. 3B\deep-learning-course-work-1-2026\data\datasets\mn_61785_test")

print(dataset)
print(dataset[0])