from train import train_unet
import json


learn_rate = [0.001]
epoch = 10
scales = [1, 0.7, 0.5, 0.3]
batches = [1, 2, 4, 8, 16, 32]
val = 11

net_test = train_unet()
best_model = {'score': 0, 'properties' : ''}
best_model['score'] = 10
list_results = {}
for lr_rate in learn_rate:
    for scale in scales:
        for batch in batches:
            output_path = f'checkpoints_fill/checkoints_LR_{lr_rate}_BS_{batch}_SCALE_{scale}_E_{epoch}/'
            try:
                val_score = net_test.train_net(
                              epochs=epoch,
                              batch_size=batch,
                              lr=lr_rate,
                              img_scale=scale,
                              val_percent=val / 100, 
                              dir_checkpoint=output_path)
                result_summary = f'model_LR_{lr_rate}_BS_{batch}_SCALE_{scale}_DICE_{val_score}_E_{epoch}\n'
                print(result_summary)
                list_results[result_summary] = val_score
                if val_score > best_model['score']:
                    best_model['score'] = val_score
                    best_model['properties'] = result_summary
            except KeyboardInterrupt:
                torch.save(net.state_dict(), 'INTERRUPTED.pth')
                logging.info('Saved interrupt')
                try:
                    sys.exit(0)
                except SystemExit:
                    os._exit(0)
print(best_model['properties'])
with open('mask_filled_results.json', 'w') as fp:
    json.dump(list_results, fp)
