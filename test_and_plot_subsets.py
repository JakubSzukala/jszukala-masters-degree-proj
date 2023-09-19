import os
from test import test_yolov7
import pandas as pd
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, help='Yolov7 model name')
    parser.add_argument('--weights', type=str, required=True, help='Path to weights .pt file')
    parser.add_argument('--nms_th', type=float, required=False, default=0.45, help='NMS threshold')
    parser.add_argument('--confidence_th', type=float, required=False, default=0.15, help='Confidence threshold for predictions')
    parser.add_argument('--dataset', type=str, required=True, help='Path to dataset descriptor file')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to dataset directory')
    parser.add_argument('--image_size', type=int, required=False, default=640, help='Image size')
    parser.add_argument('--batch_size', type=int, required=False, default=2, help='Batch size')

    cli_args = parser.parse_args()

    files_domains = [
        'Arvalis_4_train_subset.csv',
        'Arvalis_8_train_subset.csv',
        'Arvalis_3_train_subset.csv',
        'Arvalis_12_train_subset.csv',
        'Rres_1_train_subset.csv',
        'Inrae_1_train_subset.csv',
        'Arvalis_7_train_subset.csv',
        'NMBU_1_train_subset.csv',
        'NMBU_2_train_subset.csv',
        'Arvalis_2_train_subset.csv',
        'Arvalis_5_train_subset.csv',
        'Arvalis_6_train_subset.csv',
        'Arvalis_9_train_subset.csv',
        'Arvalis_10_train_subset.csv',
        'ULiège-GxABT_1_train_subset.csv',
        'Arvalis_1_train_subset.csv',
        'ETHZ_1_train_subset.csv',
        'Arvalis_11_train_subset.csv'
    ]

    files_stages = [
        'Postflowering_train_subset.csv',
        'Ripening_train_subset.csv',
        'Filling-ripening_train_subset.csv',
        'Filling_train_subset.csv',
    ]

    domain_to_devl_stage = {
        'Arvalis_1' : 'Postflowering',
        'Arvalis_2' : 'Filling',
        'Arvalis_3' : 'Filling-ripening',
        'Arvalis_4' : 'Filling',
        'Arvalis_5' : 'Filling',
        'Arvalis_6' : 'Filling-ripening',
        'Arvalis_7' : 'Filling-ripening',
        'Arvalis_8' : 'Filling-ripening',
        'Arvalis_9' : 'Ripening',
        'Arvalis_10' : 'Filling',
        'Arvalis_11' : 'Filling',
        'Arvalis_12' : 'Filling',
        'ETHZ_1' : 'Filling',
        'Inrae_1' : 'Filling-ripening',
        'NMBU_1' : 'Filling',
        'NMBU_2' : 'Ripening',
        'Rres_1' : 'Filling-ripening',
        'ULiège-GxABT_1' : 'Ripening'
    }

    df = pd.DataFrame(columns=['domain', 'stage', 'mAP05:075'])
    for domain in files_domains:
        map05_075 = test_yolov7(
            model=cli_args.model,
            weights=cli_args.weights,
            nms_th=cli_args.nms_th,
            confidence_th=cli_args.confidence_th,
            dataset=os.path.join('~/gwhd_2021-2', domain),
            data_dir=cli_args.data_dir,
            image_size=cli_args.image_size,
            batch_size=cli_args.batch_size,
        )
        domain_short = domain.split('_')[0] + '_' + domain.split('_')[1]
        stage = domain_to_devl_stage[domain_short]

        df = df.append({'domain': domain_short, 'stage': stage, 'mAP05:075': map05_075}, ignore_index=True)
        print(df)
    df.to_csv('test_results_domain.csv', index=False)
