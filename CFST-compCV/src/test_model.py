from pl_train_models import LResNet
import lightning as L
from torch.utils.data import DataLoader
from torchvision import transforms
import ds.cgqa_general as cgqa

model = LResNet.load_from_checkpoint('/home/lquarantiello/compCV/CFST-compCV/CFST-compCV/e9u6imhj/checkpoints/epoch=10-step=1727.ckpt')

# preprocessing transformations
# from CFST appendix
tr_preprocess = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
test_preprocess = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
benchmark = cgqa.continual_training_benchmark(n_experiences=10, seed=42, return_task_id=True, dataset_root='/disk3/lquara/', train_transform=tr_preprocess, eval_transform=test_preprocess)
test_loader = DataLoader(benchmark.val_datasets[0], batch_size=64, num_workers=11)

trainer = L.Trainer()
res = trainer.test(model, test_loader)