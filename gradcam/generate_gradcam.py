"""Reproduce C5 without training. Requires torch, torchvision, pandas, matplotlib, Pillow.
Usage: python generate_gradcam.py --results RESULTS_DIR --images IMAGE_DIR --out OUTPUT_DIR
Grad-CAM: Selvaraju et al. (2017), https://arxiv.org/abs/1610.02391
ResNet implementation: https://docs.pytorch.org/vision/stable/models/resnet.html
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torchvision
from torchvision import models, transforms
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def gradcam(model, x, target):
    saved = []
    hook = model.layer4.register_forward_hook(lambda module, args, output: saved.append(output))
    try:
        logits = model(x.requires_grad_(True))
        activation = saved[0]
        gradient, = torch.autograd.grad(logits[0, target], activation)
        raw = (gradient.mean((2, 3), keepdim=True) * activation).sum(1).relu()
        # Independently verify against the equivalent linear CAM for GAP + linear ResNet head.
        expected = (activation * model.fc.weight[target][None, :, None, None]).sum(1).relu() / 49
        assert torch.allclose(raw, expected, atol=1e-6, rtol=1e-4)
        assert raw.shape == (1, 7, 7) and torch.isfinite(raw).all() and raw.max() > 0
        cam = torch.nn.functional.interpolate(raw[:, None], (224, 224), mode='bilinear', align_corners=False)[0, 0]
        cam = (cam - cam.min()) / (cam.max() - cam.min()).clamp_min(1e-12)
        return cam.detach().numpy(), logits.detach().softmax(1)[0].numpy()
    finally:
        hook.remove()


def main():
    parser = argparse.ArgumentParser()
    for name in ['results', 'images', 'out']:
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    manifest = pd.read_csv(args.results / 'split_manifest.csv')
    names = ['Abyssinian_158', 'american_bulldog_100', 'boxer_182']
    runs = ['B2_aug', 'B3_progressive_aug']
    crop = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224)])
    tensor = transforms.Compose([transforms.ToTensor(), transforms.Normalize([.485, .456, .406], [.229, .224, .225])])
    model_list, metadata = [], {}
    for run in runs:
        path = args.results / run / 'best.pt'
        payload = torch.load(path, map_location='cpu', weights_only=True)
        model = models.resnet18(weights=None)
        model.fc = torch.nn.Linear(512, len(payload['classes']))
        model.load_state_dict(payload['state_dict'], strict=True)
        model.eval().requires_grad_(False)
        model_list.append(model)
        metadata[run] = {'checkpoint_sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'epoch': payload['epoch'], 'classes': payload['classes']}
    assert metadata[runs[0]]['classes'] == metadata[runs[1]]['classes']
    classes = payload['classes']
    fig, axes = plt.subplots(3, 3, figsize=(9, 8.7))
    records = []
    for i, name in enumerate(names):
        row = manifest.loc[manifest.image_id.eq(name)].iloc[0]
        image_path = args.images / (name + '.jpg')
        assert row['split'] == 'test'
        assert hashlib.sha256(image_path.read_bytes()).hexdigest() == row.sha256
        target = int(row.label)
        assert classes[target] == row.breed
        with Image.open(image_path) as im:
            input_image = crop(im.convert('RGB'))
        x = tensor(input_image).unsqueeze(0)
        axes[i, 0].imshow(input_image)
        axes[i, 0].set_title(f'{name}\nTarget: {row.breed}', fontsize=9)
        for j, (run, model) in enumerate(zip(runs, model_list), 1):
            cam, probs = gradcam(model, x.clone(), target)
            pred = int(probs.argmax())
            recorded = pd.read_csv(args.results / run / 'test_predictions.csv')
            old = recorded.loc[recorded.image_id.eq(name)].iloc[0]
            assert int(old.prediction) == pred
            assert abs(float(old.confidence) - float(probs[pred])) < 1e-4
            np.save(args.out / f'{name}_{run}_cam.npy', cam)
            axes[i, j].imshow(input_image)
            axes[i, j].imshow(cam, cmap='jet', alpha=.45, vmin=0, vmax=1)
            axes[i, j].set_title(f'{run[:2]} | pred: {classes[pred]}\nConf {100*probs[pred]:.2f}% | P(target) {100*probs[target]:.2f}%', fontsize=9)
            records.append(dict(image_id=name, target_class=row.breed, model=run, prediction=classes[pred], confidence=float(probs[pred]), target_probability=float(probs[target])))
        for ax in axes[i]:
            ax.axis('off')
    fig.tight_layout(rect=(0, .055, 1, 1))
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(0, 1), cmap='jet'), cax=fig.add_axes([.32, .035, .36, .012]), orientation='horizontal')
    cb.set_label('Grad-CAM dinormalisasi per peta (0 rendah, 1 tinggi)', fontsize=9)
    fig.savefig(args.out / 'gradcam_B2_B3.png', dpi=220, bbox_inches='tight')
    plt.close(fig)
    pd.DataFrame(records).to_csv(args.out / 'gradcam_predictions.csv', index=False)
    metadata.update(target='ground-truth class fixed across both models', layer='layer4 output, 7x7', normalization='per-map min-max after bilinear upsampling; intensity not comparable across maps', samples='three supplied test images, selected from B5 errors; not representative', comparison='B2 best epoch 25 vs B3 best epoch 8; B2 is not the exact B3 warmup checkpoint', torch=torch.__version__, torchvision=torchvision.__version__, device='CPU; eval mode; no training', source='https://arxiv.org/abs/1610.02391')
    (args.out / 'gradcam_provenance.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    print(pd.DataFrame(records).to_string(index=False))
    print('Verified image hashes, test membership, predictions, confidence, and Grad-CAM formula.')


if __name__ == '__main__':
    main()
