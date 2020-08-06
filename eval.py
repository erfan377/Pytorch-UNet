import torch
import torch.nn.functional as F
from tqdm import tqdm

from dice_loss import dice_coeff
from utils.crf import dense_crf

def eval_net(net, loader, device, crf=True):
    """Evaluation without the densecrf with the dice coefficient"""
    net.eval()
    mask_type = torch.float32 if net.n_classes == 1 else torch.long
    n_val = len(loader)  # the number of batch
    tot = 0

    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
        for batch in loader:
            imgs, true_masks = batch['image'], batch['mask']
            imgs = imgs.to(device=device, dtype=torch.float32)
            true_masks = true_masks.to(device=device, dtype=mask_type)
            
            with torch.no_grad():
                mask_pred = net(imgs)
                
            if crf:
                crf_output = np.zeros(output.shape)
                for i, (image, prob_map) in enumerate(zip(imgs, mask_pred)):
                    image = image.transpose(1, 2, 0)
                    crf_output[i] = dense_crf(image, prob_map)
                mask_pred = crf_output
                
            if net.n_classes > 1:
                tot += F.cross_entropy(mask_pred, true_masks).item()
            else:
                pred = torch.sigmoid(mask_pred)
                pred = (pred > 0.5).float()
                tot += dice_coeff(pred, true_masks).item()
            pbar.update()

    net.train()
    return tot / n_val
