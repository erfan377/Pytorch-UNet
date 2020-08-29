from train import train_unet
import json
import argparse
import csv



def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-e', '--epochs', metavar='E', type=int, nargs='*', default=[5],
                        help='Number of epochs', dest='epochs')
    parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='*', default=[1],
                        help='Batch size', dest='batchsize')
    parser.add_argument('-l', '--learning-rate', metavar='LR', type=float, nargs='*', default=[0.001],
                        help='Learning rate', dest='lr')
    parser.add_argument('-s', '--scale', dest='scale', type=float, nargs='*', default=[0.7],
                        help='Downscaling factor of the images')
    parser.add_argument('-v', '--validation', dest='val', type=float, default=11.0,
                        help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('-d', '--directory', dest='dir', type=str, default='checkpoints_test',
                        help='specify where to save the MODEL.PTH')
    parser.add_argument('-m', '--mode', dest='mod', type=str, default='normal',
                        help='specify the training mode')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    model = train_unet(args.mod)
    best_model = {'score': 0, 'properties' : ''}
    list_results = {}
    for epoch in args.epochs:
        for lr_rate in args.lr:
            for scale in args.scale:
                for batch in args.batchsize:
                    output_path = f'{args.dir}/checkoints_LR_{lr_rate}_BS_{batch}_SCALE_{scale}_E_{epoch}/'
                    try:
                        val_score = model.train_net(
                                    epochs=epoch,
                                    batch_size=batch,
                                    lr=lr_rate,
                                    img_scale=scale,
                                    val_percent=args.val / 100, 
                                    dir_checkpoint=output_path)
                        result_summary = f'model_LR_{lr_rate}_BS_{batch}_SCALE_{scale}_E_{epoch}\n'
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
    print(best_model['score'])
    with open('mask_filled_results.json', 'w') as fp:
        json.dump(list_results, fp)
    with open('mask_filled_results.csv', 'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['property', 'diceScore'])
        for key, value in list_results.items():
            writer.writerow([key, value])
