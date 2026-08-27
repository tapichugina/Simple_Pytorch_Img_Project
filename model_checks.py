
import torch
import matplotlib.pyplot as plt
import torch.optim as optim

def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")



def overfit_single_batch(model, loss_fn, train_loader, n_example=None,epochs=500, lr=1e-3, print_every=10):
    """
    Overfits a model on a single batch to sanity-check the training loop and architecture.
    
    Parameters:
    - model (nn.Module): The PyTorch model to test.
    - loss function
    - batch (tuple/list): A tuple of (inputs, targets) representing the batch.
    - epochs (int): Number of iterations to run the overfit test.
    - lr (float): Learning rate for the optimizer.
    - print_every (int): How often to print the loss.
    """
    print(f"Starting single batch overfit test for {epochs} epochs...")
    
    # Extract inputs and targets from the provided batch
    single_batch=next(iter(train_loader))

    inputs, targets = single_batch
    current_batch_size = inputs.shape[0] 
    
    if n_example is not None and n_example < current_batch_size:
        print(f"Slicing batch from {current_batch_size} down to {n_example} examples.")
        inputs = inputs[:n_example]
        targets = targets[:n_example]

    
    # Ensure model is in training mode
    model.train()
    
    # Use Adam as a reliable default for debugging
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    lossi = []
    for epoch in range(1, epochs + 1):
        # Forward pass
        optimizer.zero_grad()
        outputs = model(inputs) 
        loss = loss_fn(outputs, targets) 
        loss.backward() 
        optimizer.step()
        
        
        # Print progress
        lossi.append(loss.item())
        if epoch == 1 or epoch % print_every == 0:
            print(f"Epoch [{epoch}/{epochs}] - Loss: {loss.item():.6f}")

    with torch.no_grad():
        pred = model(inputs)
        pred_cls = pred.argmax(1)
        acc = (pred_cls == targets).float().mean()



    # --- Plotting Code ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))    
    ax1.plot(range(1, epochs + 1), lossi, label='Training Loss', color='blue', linewidth=2)
    ax1.set_title('Single Batch Overfit Test')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.legend()

    sample_indices = range(len(pred_cls))
    ax2.scatter(sample_indices, pred_cls, color='red', marker='x', s=100, label='Predicted Class', alpha=0.9)
    ax2.scatter(sample_indices, targets, color='blue', marker='o', s=50, label='True Class', alpha=0.9)
    
            
    print("Overfit test complete.\n")
    print(f"Loss: {loss.item():.6f}")
    print(f"Accuracy: {acc:.4f}")
    return loss.item()