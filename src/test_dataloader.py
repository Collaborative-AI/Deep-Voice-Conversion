from dataset import make_dataset, make_data_loader

dataset = make_dataset("MNIST")
print(dataset)
# loader = make_data_loader(dataset, 1)

# k=0
# for item in loader:
#     print(item)
#     if k > 5:
#         break
#     k+=1