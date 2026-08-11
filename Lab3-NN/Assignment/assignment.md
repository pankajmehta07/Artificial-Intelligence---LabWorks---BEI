# Assignment: Binary Classification with Neural Networks (circles dataset)

## Objective
Using the provided dataset `circles_binary_classification.csv`, reproduce and extend the workflow. Build, train, evaluate and compare PyTorch ANNs to classify the circular data and report findings.

## Dataset
- `circles_binary_classification.csv` (place in notebook folder)  

## Tasks (follow the notebook structure)
1. Data retrieval & inspection  
   - Load CSV with pandas; show `head()` and `describe()`.

2. Data cleaning & feature design  
   - Confirm/perform minimal cleaning.  
   - Create X = `[['X1','X2']]` and y = `label`.  
   - Convert to NumPy / torch tensors with correct dtypes.

3. Visualize data  
   - Scatter plot X1 vs X2 colored by label.

4. Train/test split  
   - `train_test_split(test_size=0.2, random_state=42)`.

5. Device & dtype  
   - Make code device-agnostic (`cuda` if available) and move tensors to device.

6. Implement baseline models  
   - ModelV0: 2 → 5 → 1 (no activation)  
   - ModelV1: 2 → 15 → 15 → 1 (no activation)  
   - ModelV2: 2 → 64 → 64 → 10 -> 1  with ReLU between layers

7. Loss, optimizer, metrics  
   - Use `nn.BCEWithLogitsLoss()` (raw logits).  
   - Optimizer: `torch.optim.SGD(params=model.parameters(), lr=0.1)` initially.  
   - Accuracy: round(sigmoid(logits)).

8. Training loop  
   - Use same `train_and_test_loop` pattern: training & eval each epoch, collect losses/accuracies, print progress every 10 epochs.  
   - Suggested epochs: V0 ~100, V1 ~1000, V2 ~100–1500, V3 try 500–1000 (adjust as needed).  
   - Reinitialize model & optimizer between experiments.

9. Predictions & evaluation  
    - Show untrained and trained predictions.  
    - Plot decision boundaries for train & test (use/adapt `plot_decision_boundary`).  
    - Plot train & test loss curves.

11. Discussion and Conclusion
    - compulsary

## Hints
- `BCEWithLogitsLoss` expects raw logits (do not apply sigmoid before loss).  
- Use `torch.sigmoid` on logits for probabilities and `torch.round` for predicted labels.  
- Ensure shapes match (use `squeeze()` where necessary).  
- Use `torch.manual_seed(42)` for reproducibility.  
- Reset model and optimizer before each experiment.

## Optional extra credit
- Try `Adam` vs `SGD` and compare convergence. 

## Submission
- Upload the `PDF` for the .ipynb file (might need to convert it) to the assignment portal by the deadline. Additionally, providing the exact url to the Github file in your repo is recommended. 