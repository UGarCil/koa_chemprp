from chemprop.args import TrainArgs
from chemprop.train import cross_validate
from chemprop.data import get_data, MoleculeDataset
from pathlib import Path

def train_chemprop_model():
    # Create a Path object for the output directory
    output_dir = Path('train_example')
    output_dir.mkdir(exist_ok=True)
    
    # Initialize arguments
    args = TrainArgs()
    args.from_dict({
        'data_path': './regression.csv',
        'dataset_type': 'regression',
        'save_dir': str(output_dir),
        'epochs': 30,
        'num_folds': 1,
        'metric': 'rmse',
        'target_columns': ['logSolubility']  # Specify the target column
    })
    
    # Load data
    data = get_data(path=args.data_path,
                    smiles_columns=args.smiles_columns,
                    target_columns=args.target_columns,
                    ignore_columns=args.ignore_columns,
                    skip_invalid_smiles=True)
    
    # Convert to MoleculeDataset
    data = MoleculeDataset(data)
    
    # Train model
    mean_score, std_score = cross_validate(args=args, data=data)
    
    print(f'Results:\nRMSE = {mean_score:.3f} +/- {std_score:.3f}')
    print(f'Model saved in: {output_dir}')

if __name__ == '__main__':
    train_chemprop_model()