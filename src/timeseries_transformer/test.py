import torch


print(torch.cuda.is_available())
print(torch.cuda.current_device())
print(torch.cuda.get_device_name())


acc = 0
iteration = 0
for data in test_dataloader:
  iteration += 1
  inputs, labels = data
  if torch.cuda.is_available():
    inputs = inputs.cuda()
    labels = labels.cuda()
  outputs = models[1]["model"](inputs)
  predictions = torch.argmax(outputs, axis=1)
  correct_labels = labels.squeeze().int()

  acc += (predictions == correct_labels).int().sum()/len(labels) * 100
print(acc/iteration)