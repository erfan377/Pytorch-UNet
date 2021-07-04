###############################################################
# Runs training based on list of paramaters given by the user #
###############################################################

from train import train_unet
import json
import argparse
import csv
<<<<<<< HEAD
import shutil
import os
=======
import os
import shutil
>>>>>>> cf1d56122735f32408ef93d41cbdc996f24ef3de

def get_args():
    """ Define the arguments that user can put in as flags in the terminal

    Returns:
        list: The list of inputs attached to args paramaters  
    """
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
<<<<<<< HEAD
    parser.add_argument('-e', '--epochs', metavar='E', type=int, nargs='*', default=[5],
                        help='Number of epochs', dest='epochs')
    parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='*', default=[2],
=======
    parser.add_argument('-e', '--epochs', metavar='E', type=int, nargs='*', default=[1],
                        help='Number of epochs', dest='epochs')
    parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='*', default=[1],
>>>>>>> cf1d56122735f32408ef93d41cbdc996f24ef3de
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
    """ Run by terminal to test and search for a list paramatares producing 
    best results
  
    """
    args = get_args()
<<<<<<< HEAD
    model = train_unet()
=======
    model = train_unet(args.mod)
>>>>>>> cf1d56122735f32408ef93d41cbdc996f24ef3de
    best_model = {'score': 0, 'properties' : ''}
    list_results = {} #store the result of training based on different paramaters
    for epoch in args.epochs:
        for lr_rate in args.lr:
            for scale in args.scale:
                for batch in args.batchsize:
                    output_path = f'{args.dir}/checkoints_LR_{lr_rate}_BS_{batch}_SCALE_{scale}_E_{epoch}/'
<<<<<<< HEAD
                    if os.path.isdir(output_path):
                        shutil.rmtree(output_path) 
                    os.makedirs(output_path)
=======
                    if os.path.exists(output_path):
                        shutil.rmtree(output_path)
>>>>>>> cf1d56122735f32408ef93d41cbdc996f24ef3de
                    val_score = model.train_net(
                                epochs=epoch,
                                batch_size=batch,
                                lr=lr_rate,
                                img_scale=scale,
<<<<<<< HEAD
                                augment=False,
=======
>>>>>>> cf1d56122735f32408ef93d41cbdc996f24ef3de
                                val_percent=args.val / 100, 
                                dir_checkpoint=output_path)

                    result_summary = f'model_LR_{lr_rate}_BS_{batch}_SCALE_{scale}_E_{epoch}\n'
                    list_results[result_summary] = val_score
                    if val_score > best_model['score']:
                        best_model['score'] = val_score
                        best_model['properties'] = result_summary
<<<<<<< HEAD
#                     except KeyboardInterrupt:
#                         torch.save(net.state_dict(), 'INTERRUPTED.pth')
#                         logging.info('Saved interrupt')
#                         try:
#                             sys.exit(0)
#                         except SystemExit:
#                             os._exit(0)
=======

>>>>>>> cf1d56122735f32408ef93d41cbdc996f24ef3de
    print(best_model['properties'])
    print(best_model['score'])
    #store the training in 2 formats of json and CSV
    with open('mask_filled_results.json', 'w') as fp:
        json.dump(list_results, fp)
    with open('mask_filled_results.csv', 'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['property', 'diceScore'])
        for key, value in list_results.items():
            writer.writerow([key, value])
