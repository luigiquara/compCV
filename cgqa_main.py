from cfst import cgqa
from cfst import resnet
import torch
import argparse
import numpy as np
from torch.nn import CrossEntropyLoss
import avalanche as avl
from avalanche.training import Naive
from avalanche.training.plugins import EvaluationPlugin, EarlyStoppingPlugin, LRSchedulerPlugin
from avalanche.evaluation import metrics as metrics


parser = argparse.ArgumentParser()
parser.add_argument('--n_experiences_continual', type=int, default=10)
parser.add_argument('--n_experiences_fewshot', type=int, default=300)
parser.add_argument('--fewshot_classes', type=int, default=10)
parser.add_argument('--mode', type=str, default='sys',
                    choices=['sys', 'pro', 'sub', 'non', 'noc', 'all'])
parser.add_argument('--use_task_id', action="store_true")
parser.add_argument('--no_cuda', action="store_true")
parser.add_argument('--root', type=str, default='/raid/a.cossu/datasets/CFST')
args = parser.parse_args()

modes = [args.mode] if args.mode != 'all' else ['sys', 'pro', 'sub', 'non', 'noc']
device = 'cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu'

continual_benchmark = cgqa.continual_training_benchmark(
    n_experiences=args.n_experiences_continual,
    image_size=(128, 128),
    return_task_id=args.use_task_id,
    seed=None,
    fixed_class_order=None,
    shuffle=True,
    train_transform=None,
    eval_transform=None,
    dataset_root=args.root,
    memory_size=0,
    num_samples_each_label=None,
    multi_task=False)  # multi-task aka joint training not working properly

# either multi-head or incremental classifier resnet18
model = resnet.get_resnet(multi_head=args.use_task_id,
                          initial_out_features=10)

loggers = [
    avl.logging.InteractiveLogger(),
    avl.logging.TensorboardLogger()
]

metrics_list = [
    metrics.accuracy_metrics(epoch=True, experience=True, stream=True),
    metrics.loss_metrics(epoch=True, experience=True, stream=True),
    metrics.forgetting_metrics(experience=True, stream=True),
    metrics.timing_metrics(epoch=True)]

evaluation_plugin = EvaluationPlugin(
    *metrics_list,
    loggers=loggers)

optimizer = torch.optim.Adam(model.parameters(), lr=8e-3)
epochs = 1
plugins = [
    LRSchedulerPlugin(scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs, 1e-6)),
    EarlyStoppingPlugin(patience=5, val_stream_name='val_stream')
]
strategy = Naive(model, optimizer, CrossEntropyLoss(),
                 train_mb_size=100, train_epochs=epochs, eval_mb_size=50, device=device,
                 plugins=plugins, evaluator=evaluation_plugin, eval_every=1, peval_mode="epoch")

results = []
for exp_idx, (experience, val_task) in enumerate(zip(continual_benchmark.train_stream, continual_benchmark.val_stream)):
    print("Start of experience ", experience.current_experience)
    print("Current Classes: ", experience.classes_in_this_experience)
    print("Current Classes (str): ", [
        continual_benchmark.label_info[2][cls_idx]
        for cls_idx in continual_benchmark.original_classes_in_exp[experience.current_experience]
    ])

    strategy.train(experience, eval_streams=[val_task])
    print("Training experience completed")

    print("Computing accuracy on the whole test set.")
    result = strategy.eval(continual_benchmark.test_stream)
    results.append(result)

    if exp_idx == 1:
        break

# few-shot test
size_before_classifier = model.classifier.in_features if args.use_task_id else model.classifier.classifier.in_features

# Freeze the parameters of the feature extractor
for param in model.resnet.parameters():
    param.requires_grad = False

for mode in modes:
    print(f'mode: {mode}')

    fewshot_benchmark = cgqa.fewshot_testing_benchmark(
        n_experiences=args.n_experiences_fewshot,
        image_size=(128, 128),
        n_way=args.fewshot_classes,
        n_shot=10,
        n_val=5,
        n_query=10,
        mode=mode,
        task_offset=args.n_experiences_continual if not args.use_task_id else 1,
        seed=None,
        fixed_class_order=None,
        train_transform=None,
        eval_transform=None,
        dataset_root=args.root)

    loggers = [avl.logging.InteractiveLogger()]

    metrics_list = [
        metrics.accuracy_metrics(epoch=True, experience=True, stream=True),
        metrics.loss_metrics(epoch=True, experience=True, stream=True)
    ]

    evaluation_plugin = EvaluationPlugin(
        *metrics_list,
        loggers=loggers)

    # Start training
    results = []
    for exp_idx, experience in enumerate(fewshot_benchmark.train_stream):
        print("Start of experience ", experience.current_experience)
        print("Current Classes: ", experience.classes_in_this_experience)
        print("Current Classes (str): ", [
            fewshot_benchmark.label_info[2][cls_idx]
            for cls_idx in fewshot_benchmark.original_classes_in_exp[experience.current_experience]
        ])

        model.classifier = torch.nn.Linear(size_before_classifier, args.fewshot_classes)

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        epochs = 20
        plugins = [
            LRSchedulerPlugin(scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs, 1e-6))
        ]
        strategy = Naive(model, optimizer, CrossEntropyLoss(),
                         train_mb_size=100, train_epochs=epochs, device=torch.device("cuda:0"),
                         plugins=plugins, evaluator=evaluation_plugin)

        strategy.train(experience)
        print("Training experience completed")

        print("Computing accuracy.")
        result = strategy.eval(fewshot_benchmark.test_stream[experience.current_experience])
        results.append(result)

    print(results)
