import os
import argparse
import pandas as pd
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from dataset import *
from model import *

path_arr = [
    './dataset/city_A_challengedata.csv.gz',
    './dataset/city_B_challengedata.csv.gz',
    './dataset/city_C_challengedata.csv.gz',
    './dataset/city_D_challengedata.csv.gz'
]

city_names = ['A', 'B', 'C', 'D']

def Test(args):
    """
    Performs inference on the final test set for specified cities.
    The test set contains 3000 users for cities B, C, and D.
    Note: The TestSet class does not provide ground-truth labels.
    """
    # Determine which cities to process
    if args.cities:
        city_indices = []
        for city in args.cities:
            city_upper = city.upper()
            if city_upper in ['B', 'C', 'D']:
                city_indices.append(city_names.index(city_upper))
            else:
                print(f"Warning: Test set is only available for cities B, C, and D. Skipping city {city}.")
        
        if not city_indices:
            print("No valid cities specified for testing. Exiting.")
            return
    else:
        # Default to cities B, C, D (indices 1, 2, 3)
        city_indices = [1, 2, 3]
        print("No cities specified. Processing default cities: B, C, D")
    
    # Process each city
    for city_idx in city_indices:
        city_letter = city_names[city_idx]
        print(f"\n=== Processing City {city_letter} (Test Set) ===")
        
        # Set the storage path for results
        result_path = f'/content/drive/MyDrive'
        generated_name = f'city{city_letter}_test_generated.csv.gz'
        os.makedirs(result_path, exist_ok=True)
        
        # Load the test set using the TestSet class
        # Note: TestSet does not require the is_100val parameter
        dataset_test = TestSet(path_arr[city_idx])
        dataloader_test = DataLoader(dataset_test, batch_size=1, num_workers=args.num_workers)
        
        # Specify the GPU device
        device = torch.device(f'cuda:{args.cuda}')
        
        # Instantiate the model and load the pre-trained parameters
        model = LPBERT(args.layers_num, args.heads_num, args.embed_size, args.city_embed).to(device)
        model.load_state_dict(torch.load(args.pth_file, map_location=device))
        
        # Initialize a list to store all generated predictions
        all_generated = []
        
        # Set the model to evaluation mode
        model.eval()
        
        with torch.no_grad():
            for data in tqdm(dataloader_test, desc=f"City {city_letter}"):
                # Load data to GPU
                data['uid'] = data['uid'].to(device)
                data['d'] = data['d'].to(device)
                data['t'] = data['t'].to(device)
                data['input_x'] = data['input_x'].to(device)
                data['input_y'] = data['input_y'].to(device)
                data['time_delta'] = data['time_delta'].to(device)
                data['city'] = data['city'].to(device)
                data['len'] = data['len'].to(device)
                
                # Get the predictions
                output = model(data['d'], data['t'], data['input_x'], data['input_y'], data['time_delta'], data['len'], data['city'])
                
                # Process the output
                assert torch.all((data['input_x'] == 201) == (data['input_y'] == 201))
                pred_mask = (data['input_x'] == 201)
                output = output[pred_mask]
                
                # This prediction logic is the same as in your validation script
                pred = []
                pre_x, pre_y = -1, -1
                for step in range(len(output)):
                    if step > 0:
                        output[step][0][pre_x] *= 0.9
                        output[step][1][pre_y] *= 0.9
                    pred.append(torch.argmax(output[step], dim=-1))
                    pre_x, pre_y = pred[-1][0].item(), pred[-1][1].item()
                
                # Generate prediction results (uid, d, t, x, y)
                pred = torch.stack(pred)
                generated = torch.cat((data['uid'][pred_mask].unsqueeze(-1),
                                     data['d'][pred_mask].unsqueeze(-1)-1,
                                     data['t'][pred_mask].unsqueeze(-1)-1,
                                     pred+1), dim=-1).cpu().tolist()
                
                # Add the generated points to the list
                for point in generated:
                    all_generated.append(point)
        
        # Convert the list to a DataFrame and save
        columns = ['uid', 'd', 't', 'x', 'y']
        generated_df = pd.DataFrame(all_generated, columns=columns)
        
        # Save the CSV file (compressed)
        generated_df.to_csv(os.path.join(result_path, generated_name), index=False, compression='gzip')
        
        print(f"Generated test predictions saved to: {os.path.join(result_path, generated_name)}")
        print(f"Total generated points: {len(all_generated)}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pth_file', type=str, default='/content/drive/MyDrive/best_finetune_model.pth')
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--embed_size', type=int, default=128)
    parser.add_argument('--city_embed', type=int, default=4)
    parser.add_argument('--layers_num', type=int, default=4)
    parser.add_argument('--heads_num', type=int, default=8)
    parser.add_argument('--cuda', type=int, default=0)
    parser.add_argument('--cities', nargs='*', type=str,
                       help='Cities to process (e.g., --cities B C D). If not specified, defaults to B C D')
    
    args = parser.parse_args()
    Test(args)
