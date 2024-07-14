_base_ = './1-ORCNN_pr_r50_dota_ms.py'

# model
model = dict(pretrained='torchvision://resnet101', backbone=dict(depth=101))
