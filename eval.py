import csv
import torch
from model.dataset import EvalImageGroundTruthFolder
from model.evaluate import Eval_thread
from torchvision import transforms

dataset = EvalImageGroundTruthFolder('./datasets/small', './results/small', transform=transforms.ToTensor(), target_transform=transforms.ToTensor())
loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

eval = Eval_thread('./datasets/small', loader, method='MAE', dataset='Test', output_dir='./')
results = eval.run()

for d in results:
    print(d)
    for m, r in results[d].items():
        print(m, r)
    with open(d+'.csv', 'w') as outfile:
        writer = csv.writer(outfile)
        for key, val in results[d].items():
            writer.writerow([key, val])
    print()
