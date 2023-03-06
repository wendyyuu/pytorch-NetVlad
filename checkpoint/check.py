import torch
import matplotlib.pyplot as plt
# import torchvision.models as models

#                         'epoch': epoch,
#                         'state_dict': model.state_dict(),
#                         'recalls': recalls,
#                         'best_score': best_score,
#                         'optimizer' : optimizer.state_dict(),
#                         'parallel' : isParallel,
#                         'recall_history':recall_history,


checkpoint = torch.load('/media/sdc1/wendyu/ibrahimi/runs/Feb19_03-46-01_vgg16_netvlad/checkpoints/checkpoint.pth.tar')
# model_best = torch.load('model_best.pth.tar')
# for key in checkpoint['recalls']:
#     # print("checkpoint's recalss = ", checkpoint['recalls'])
#     print("checkpoint's epoch = ", checkpoint['epoch'])
#     print("checkpoint's recall_history = ", checkpoint['recall_history'])
#     # print("model_best's recalss = ", model_best['recalls'])
#     # print("model_best's epoch = ", model_best['epoch'])

# for key in checkpoint.keys():
#     print(key)

print("checkpoint's epoch = ", checkpoint['epoch'])
print("checkpoint's recall_history = ", checkpoint['recall_history'])



x1points = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

y1points = []
y2points = []
y3points = []
y4points = []
for dict_key, dict_val in checkpoint['recall_history'].items():
    # for key, val in dict_val.items():
    # print(dict_key)
    y1points.append(dict_val[1])
    y2points.append(dict_val[5])
    y3points.append(dict_val[10])
    y4points.append(dict_val[20])

# print(y1points)

plt.plot(x1points, y1points, label = "Recall @ 1")
plt.plot(x1points, y2points, label = "Recall @ 5")
plt.plot(x1points, y3points, label = "Recall @ 10")
plt.plot(x1points, y4points, label = "Recall @ 20")
plt.legend(loc="lower right")
plt.xlabel("Epoch")
plt.ylabel("Recall Rate")
plt.title("Recall @ N in epoch")

plt.show()
plt.savefig("test")
