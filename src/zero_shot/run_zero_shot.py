import os
import time

pr_sets = ['imagenet', 'cifar10', 'cifar100']
#task_datasets = ['oxford102flowers', 'stanford_cars', 'inaturalist']
task_datasets = ['stanford_cars']

start = time.time()
for pr in pr_sets:
    for ds in task_datasets:
        os.system("python zero_shot.py --model_name resnet50 --pretraining_set {} --task_dataset {} --head logistic_regression --log_file zero_shot_400iter_acc.out".format(pr, ds))
print(f'Done in {time.time()-start} seconds')
