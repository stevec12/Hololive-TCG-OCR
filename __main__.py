import argparse
import download
import getframes
import initialFilterTrain
import initialFilterPreds
import secondFilterTrain
import secondFilterPreds
import imgOCRfinetune
import imgOCRtransformer
import predCheck
import sys

def main() -> int:
    parser = argparse.ArgumentParser(
        prog='StreamOCR',
        description='Supports the OCR of YouTube streams.' +
        '\n Requires an associated Excel file in the same directory including Channel Name, Channel ID, Video ID(s)',
        epilog='Created by: Olmen'
        )
    parser.add_argument('-d', '--download', type=bool, default=False, help='Whether to download associated videos.')
    parser.add_argument('--excel_path', type=str, default=None, help='Excel path storing Member_Name|Channel ID|VID(s)')
    parser.add_argument('name', type=str, help='The YouTube Channel Name, matching the included Excel file.')
    parser.add_argument('-g', '--getframes', type=bool, default=False, help='Whether to get the frames from videos.')  
    parser.add_argument('-fs', '--frameskip', type=int, default=1, help='How many seconds of frames to skip between captures.')
    parser.add_argument('--img_dim', nargs=2, type=int, default=[360,640], help='Image [height,width] of captures.')
    parser.add_argument('-t', '--threads', type=int, default=2, help='Number of threads to use for multiprocessing.')
    parser.add_argument('-ti', '--train_initial', type=bool, default=False, help='Whether to train the initial filter.')
    parser.add_argument('-initial_filter_path', type=str, default=None, help="Folder containing sorted initial filter training images.")
    parser.add_argument('-ts', '--train_second', type=bool, default=False, help='Whether to train the second filter.')
    parser.add_argument('-second_filter_path', type=str, default=None, help='Folder containing sorted second filter training images.')
    parser.add_argument('--bd_dim', nargs=4, type=int, default=[260,460,30,120], help='Crop bounding box bottom-left corner and height-width.')
    
    parser.add_argument('-fi', '-filtering', type=bool, default=True, help='Whether to filter the captures.')
    
    parser.add_argument('-ft', '-finetune', type=bool, default=True, help='Whetehr to fine-tune the OCR processor.')
    parser.add_argument('-tm', '--transformer_model', type=str, default="microsoft/trocr-small-printed", help="The Transformers model name to use for OCR.")
    parser.add_argument('--ocr_pretrain_root', type=str, default='D:/Side Projects D/repo/TCG Shop Stream OCR/Solution1/OCR Pretrains/imgs/', 
                        help="Root for where the OCR pretrain images are located.")
    parser.add_argument('--ocr_pretrain_labels', type=str, default="OCR Pretrains/imgs_labels.txt", 
                        help="Relative path to the OCR pretrain image labels.")
    parser.add_argument('-dv', '--device', type=str, default='cpu', help="Device on which to pretrain OCR ('cpu','cuda').")
    parser.add_argument('-tl', '--target_length', type=int, default=10, help="String max target length for OCR.")
    
    args = parser.parse_args()
    
    # Download videos if needed
    if args['-d']==True:
        download.vidDownload(args['excel_path'], args['name'])
    # Get frames from the videos
    if args['-g']==True:
        getframes.main(args.excel_path, args.name, args.threads)
    # Train initial filter
    if args['ti']==True:
        initialFilterTrain.main(args.initial_filter_path, args.img_dim[0], args.img_dim[1])
    # Train second filter
    if args['ts']==True: 
        secondFilterTrain.main(args.second_filter_path, args.img_dim[0], args.img_dim[1],
                               args.bd_dim[0],args.bd_dim[1],args.bd_dim[2],args.bd_dim[3])
    
    # Perform filtering 
    # Note: must be done at least once before moving to next OCR steps to ensure files in right paths
    if args['fi']==True:
        initialFilterPreds.main(args.name, args.img_dim[0], args.img_dim[1])
        secondFilterPreds.main(args.name, args.img_dim[0], args.img_dim[1],
                               args.bd_dim[0],args.bd_dim[1],args.bd_dim[2],args.bd_dim[3])
    
    # Fine-tune OCR processor
    if args['ft']==True:
        imgOCRfinetune.main(args.transformer_model, args.ocr_pretrain_root, args.ocr_pretrain_labels,
                            args.device, args.target_length,
                            args.bd_dim[1],args.bd_dim[0], args.bd_dim[1]+args.bd_dim[3], args.bd_dim[0]+args.bd_dim[2])

    # OCR predictions
    imgOCRtransformer.main(args.name, args.processor, args.img_dim[0], args.img_dim[1].
                           args.bd_dim[0],args.bd_dim[1],args.bd_dim[2],args.bd_dim[3])
    # Prediction checkin
    predCheck.main(args.name)
    return 0

if __name__ == '__main__':
    main()
    sys.exit()