from chemprop.train import predict
from chemprop.data import MoleculeDataset, MoleculeDataLoader
from chemprop.utils import load_args, load_checkpoint, load_scalers

def predict_properties(smiles_list, model_path):
    """
    Predict properties for a list of SMILES strings using a trained Chemprop model.
    
    Args:
        smiles_list (list): List of SMILES strings
        model_path (str): Path to the trained model checkpoint
    
    Returns:
        list: Predicted properties for each molecule
    """
    # Load trained model
    args = load_args(model_path)
    model = load_checkpoint(model_path, device='cuda' if args.cuda else 'cpu')
    scaler = load_scalers(model_path)
    
    # Create dataset and data loader
    test_data = MoleculeDataset([{'smiles': [smile]} for smile in smiles_list])
    test_loader = MoleculeDataLoader(
        dataset=test_data,
        batch_size=50,
        num_workers=0
    )
    
    # Make predictions
    preds = predict(
        model=model,
        data_loader=test_loader,
        scaler=scaler
    )
    
    return preds

if __name__ == '__main__':
    # Example usage
    model_path = 'path/to/your/model.pt'
    smiles_list = [
        'CC(=O)OC1=CC=CC=C1C(=O)O',  # Aspirin
        'CN1C=NC2=C1C(=O)N(C(=O)N2C)C'  # Caffeine
    ]
    
    predictions = predict_properties(smiles_list, model_path)
    
    # Print results
    for smile, pred in zip(smiles_list, predictions):
        print(f'SMILES: {smile}')
        print(f'Predictions: {pred}\n')
        
        
# chemprop