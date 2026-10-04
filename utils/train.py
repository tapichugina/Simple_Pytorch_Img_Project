import numpy as np
import pandas as pd
from datetime import datetime
import torch
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import f1_score, confusion_matrix, classification_report



class StepByStep(object):
    def __init__(self, model, loss_fn, optimizer,scheduler=None):
        # Here we define the attributes of our class
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # Let's send the model to the specified device right away
        self.model.to(self.device)

        # These attributes are defined here, but since they are
        # not informed at the moment of creation, we keep them None
        self.train_loader = None
        self.val_loader = None
        self.writer = None
        
        # These attributes are going to be computed internally
        self.train_losses = []
        self.val_losses = []
        self.lr_log=[]

        self.val_report=[]
        self.total_epochs = 0


    def to(self, device):
        # This method allows the user to specify a different device
        # It sets the corresponding attribute (to be used later in
        # the mini-batches) and sends the model to the device
        try:
            self.device = device
            self.model.to(self.device)
        except RuntimeError:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            print(f"Couldn't send it to {device}, sending it to {self.device} instead.")
            self.model.to(self.device)

    def set_loaders(self, train_loader, val_loader=None):
        # This method allows the user to define which train_loader (and val_loader, optionally) to use
        # Both loaders are then assigned to attributes of the class
        # So they can be referred to later
        self.train_loader = train_loader
        self.val_loader = val_loader

    def set_tensorboard(self,name, folder='runs'):
        # This method allows the user to define a SummaryWriter to interface with TensorBoard
        current_time = datetime.now().strftime("%b%d_%H-%M-%S")
        self.writer = SummaryWriter(f"{folder}/{current_time}_{name}")
        


    def _train_step(self,x,y):
        self.model.train()
        self.optimizer.zero_grad()
        # Step 1 - Computes our model's predicted output - forward pass
        yhat = self.model(x)
        # Step 2 - Computes the loss
        loss = self.loss_fn(yhat, y)
        # Step 3 - Computes gradients for both "a" and "b" parameters
        loss.backward()
        # Step 4 - Updates parameters using gradients and the learning rate
        self.optimizer.step()

        # Returns the loss
        return loss.item()

    def _val_step(self,x,y):
        self.model.eval()
        # Step 1 - Computes our model's predicted output - forward pass
        with torch.no_grad():
            yhat = self.model(x)
            pred_class= torch.argmax(yhat, dim=1)
            # Step 2 - Computes the loss
            loss = self.loss_fn(yhat, y)

        return loss.item(),pred_class
    
    def _classification_report(self,epoch_targets,epoch_preds):
        f1 = f1_score(epoch_targets, epoch_preds, average="macro")
        cm=confusion_matrix(epoch_targets,epoch_preds)
        c_report=classification_report(epoch_targets, epoch_preds, zero_division=0)
        return {"f1_score": f1,
                "cm": cm,
                "c_report": c_report}

            
    def _train_mini_batch(self):
        mini_batch_losses = []
        for x_batch, y_batch in self.train_loader:
            x_batch = x_batch.to(self.device)
            y_batch = y_batch.to(self.device)

            mini_batch_loss =self._train_step(x_batch, y_batch)
            mini_batch_losses.append(mini_batch_loss)

        loss = np.mean(mini_batch_losses)
        return loss
    
    def _val_mini_batch(self):
        mini_batch_losses = []
        all_preds=[]
        all_targets=[]
        for x_batch, y_batch in self.val_loader:
            x_batch = x_batch.to(self.device)
            y_batch = y_batch.to(self.device)

            mini_batch_loss,pred_class =self._val_step(x_batch, y_batch)
            mini_batch_losses.append(mini_batch_loss)
    
            all_preds.extend(pred_class.cpu().numpy())
            all_targets.extend(y_batch.cpu().numpy())
        report=self._classification_report(all_targets,all_preds)
            
        loss = np.mean(mini_batch_losses)
        
        return loss,report

    def set_seed(self, seed=42):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False    
        torch.manual_seed(seed)
        np.random.seed(seed)
    
    def train(self, n_epochs, seed=42):
        # To ensure reproducibility of the training process
        self.set_seed(seed)

        for epoch in range(n_epochs):
            # Keeps track of the numbers of epochs
            # by updating the corresponding attribute
            self.total_epochs += 1

            # inner loop
            # Performs training using mini-batches
            loss = self._train_mini_batch()
            self.train_losses.append(loss)

            # VALIDATION
            # no gradients in validation!
            with torch.no_grad():
                # Performs evaluation using mini-batches
                val_loss,report = self._val_mini_batch()
                self.val_losses.append(val_loss)
                self.val_report.append(report)

            # SCHEDULER
            if self.scheduler is not None:
                self.scheduler.step()    


            if epoch % 1 == 0:
                status_str = f"Epoch [{epoch}/{n_epochs}] -> Train Loss: {loss:.4f}"
                if val_loss is not None:
                    status_str += f" | Val Loss: {val_loss:.4f} | F1_ score: {report['f1_score']:.4f}"
                print(status_str)

            # If a SummaryWriter has been set...
            if self.writer:
                scalars = {'training': loss,'validation': val_loss}
                # Records both losses for each epoch under the main tag "loss"
                self.writer.add_scalars(main_tag='loss',
                                        tag_scalar_dict=scalars,
                                        global_step=epoch)
                current_lr = self.optimizer.param_groups[0]['lr']
                self.lr_log.append(current_lr)
                self.writer.add_scalar('learning_rate',current_lr,global_step=epoch)
                 

                for name, param in self.model.named_parameters():
                    self.writer.add_histogram(name, param, global_step=epoch)
                    # Parameter mean and std
                    self.writer.add_scalar(f"parameters/{name}/mean",param.mean().item(),global_step=epoch)
                    self.writer.add_scalar(f"parameters/{name}/std",param.std().item(),global_step=epoch)

                    if param.grad is not None:
                        self.writer.add_histogram(f"{name}.grad", param.grad, global_step=epoch)
                        self.writer.add_scalar(f"parameters/{name}.grad/mean",param.grad.mean().item(),global_step=epoch)
                        self.writer.add_scalar(f"parameters/{name}.grad/std",param.grad.std().item(),global_step=epoch)



        if self.writer:
            self.writer.add_text("Model/architecture",str(self.model),global_step=0)
            self.writer.add_text(
                "Training/config",f"""
                Optimizer: {self.optimizer.__class__.__name__}
                Learning rate: {self.optimizer.param_groups[0]['lr']}
                Weight decay: {self.optimizer.param_groups[0]['weight_decay']}""",
                global_step=0)
            # Closes the writer
            self.writer.close()

    def save_checkpoint(self, filename):
        # Builds dictionary with all elements for resuming training
        checkpoint = {'epoch': self.total_epochs,
                      'model_state_dict': self.model.state_dict(),
                      'optimizer_state_dict': self.optimizer.state_dict(),
                      'loss': self.train_losses,
                      'val_loss': self.val_losses}

        torch.save(checkpoint, filename)

    def load_checkpoint(self, filename):
        # Loads dictionary
        checkpoint = torch.load(filename, weights_only=False)

        # Restore state for model and optimizer
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        self.total_epochs = checkpoint['epoch']
        self.losses = checkpoint['loss']
        self.val_losses = checkpoint['val_loss']

        self.model.train() # train for resuming training  

    def predict(self, x):
        # Set is to evaluation mode for predictions
        self.model.eval() 
        with torch.no_grad():
            # Takes aNumpy input and make it a float tensor
            x_tensor = torch.as_tensor(x).float()
            # Send input to device and uses model for prediction
            y_hat_tensor = self.model(x_tensor.to(self.device))
        return y_hat_tensor.detach().cpu().numpy()

    def plot_losses(self):
        fig = plt.figure(figsize=(10, 4))
        plt.plot(self.train_losses, label='Training Loss', c='b')
        plt.plot(self.val_losses, label='Validation Loss', c='r')
        plt.yscale('log')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.tight_layout()
        return fig

    def add_graph(self):
        # Fetches a single mini-batch so we can use add_graph
        if self.train_loader and self.writer:
            x_sample, y_sample = next(iter(self.train_loader))
            self.writer.add_graph(self.model, x_sample.to(self.device))

    def attach_hooks(self, layers_to_hook):
        self.layers_to_hook = layers_to_hook
        self.activations = {}
        self.activations_grad ={}
        self.handles = {}

        def get_activation(name):
            def hook(layer, inputs, outputs):
                #print(f"{name} hook called, shape = {outputs.shape}")
                # Save activation
                self.activations[name].append(outputs.detach().cpu())

                # Save gradient flowing through activation
                if outputs.requires_grad:
                    outputs.register_hook(
                        lambda grad , name=name: self.activations_grad[name].append(grad.detach().cpu())
                        )
            return hook

        for layer, module in self.model.named_modules():
            if layer in layers_to_hook:

                self.activations[layer] = []
                self.activations_grad[layer] = []

                self.handles[layer] = module.register_forward_hook(
                    get_activation(layer)
                )

        #print(self.handles)

    def capture_gradients(self, layers_to_hook):
        if not isinstance(layers_to_hook, list):
            layers_to_hook = [layers_to_hook]

        self._gradients = {}
            
        def make_log_fn(name, parm_id):
            def log_fn(grad):
                self._gradients[name][parm_id].append(grad.detach().clone())
                return
            return log_fn

        for name, layer in self.model.named_modules():
            if name in layers_to_hook:
                self._gradients[name]={}
                for parm_id, p in layer.named_parameters():
                    if p.requires_grad:
                        self._gradients[name][parm_id]=[]
                        log_fn = make_log_fn(name, parm_id)
                        self.handles[f'{name}.{parm_id}.grad'] = p.register_hook(log_fn)
        return

    def _save_parameters_before_update(self):
        """Save a copy of parameters before optimizer.step()."""
    
        self._params_before_update = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self._params_before_update[name]=param.detach().clone()

    def _calculate_relative_updates(self):
        """Calculate ||ΔW|| / ||W|| for every parameter."""
        relative_updates = {}

        for name, param in self.model.named_parameters():
            if param.requires_grad:
                W_before = self._params_before_update[name]
                W_after = param.detach()
                # ΔW = W_after - W_before
                delta_W = W_after - W_before
                # ||ΔW||
                update_norm = delta_W.norm()
                # ||W||
                parameter_norm = W_before.norm()
                # ||ΔW|| / ||W||
                relative_update = (
                    update_norm / (parameter_norm + 1e-12)
                )
                relative_updates[name] = relative_update.item()

        return relative_updates


    def diagnostic_batch(self, x, y):
        
        # clear previous probe results
        for name in self.activations:
            self.activations[name] = []
            self.activations_grad[name] = []

        self.model.eval()
        self.optimizer.zero_grad()

        x = x.to(self.device)
        y = y.to(self.device)

        yhat = self.model(x)
        loss = self.loss_fn(yhat, y)
        loss.backward()

        self._save_parameters_before_update()
        self.optimizer.step()
        relative_updates = self._calculate_relative_updates()

        self.load_checkpoint("model_probe.pth")


        stats = []
        for name in self.activations:

            inspect_layer = self.activations[name][0]
            inspect_layer_grad = self.activations_grad[name][0]

            layer_stat = {
                "layer": name,
                "shape": tuple(inspect_layer.shape),
                "mean": inspect_layer.mean().item(),
                "std": inspect_layer.std().item(),
                "min": inspect_layer.amin().item(),
                "max": inspect_layer.amax().item(),
                "zero_fraction": (inspect_layer <= 0).float().mean().item(),
                "mean_grad": inspect_layer_grad.abs().mean().item(),
                "grad_std": inspect_layer_grad.std().item(),
                "weight_relative_update":relative_updates.get(f"{name}.weight", None),
                "bias_relative_update":relative_updates.get(f"{name}.bias", None)
            }

            stats.append(layer_stat)

        df = pd.DataFrame(stats)
        return df

    def diagnostic_epoch(self, train_loader,layers_to_hook):
        self.save_checkpoint("model_probe.pth")
        self.attach_hooks(layers_to_hook)
        epoch_stats = []
        for batch_idx, (x, y) in enumerate(train_loader):
            batch_df = self.diagnostic_batch(x, y)
            batch_df["batch"] = batch_idx
            epoch_stats.append(batch_df)


        self.remove_hooks()
        # this is nessesary while the probe used optimizerstep and updates the model
        self.load_checkpoint("model_probe.pth")    

        return pd.concat(epoch_stats, ignore_index=True)
    
    def summarize_epoch(self,epoch_df):
        summary = (
            epoch_df
            .groupby("layer")
            .agg({
                "mean": "mean",
                "std": "mean",
                "min": "min",
                "max": "max",
                "zero_fraction": "mean",
                "mean_grad": "mean",
                "grad_std": "mean",
                "weight_relative_update": "mean",
                "bias_relative_update": "mean"
            })
            .reset_index()
        )
        return summary
    


    def remove_hooks(self):
        for handle in self.handles.values():
            handle.remove()
        self.handles = {}


    def _visualize_tensor(self,x, note="",filters=None):
        # if x.ndim == 2:
        #     x = x.unsqueeze(0)

        # Turns (N,) into (1, 1, N) or (H, W) into (1, H, W) in one step
        while x.ndim < 3:
            x = x.unsqueeze(0)  # PyTorch

        x=x.numpy()
        # x = np.expand_dims(x, 0) # NumPy equivalent
        # Normalize if values are PyTorch tensor
        x_min, x_max = x.min(), x.max()
        x = (x - x_min) / (x_max - x_min + 1e-8)

        #print(x.shape)
        
        
        if filters==True:
            if x.ndim != 4:
                raise ValueError(f"Expected 4D tensor (N,C,H,W), got {x.shape}")

            n_filters=x.shape[0]
            num_channels = x.shape[1]
            height=x.shape[2]/10
            width=x.shape[3]/10

            fig, ax = plt.subplots(n_filters,num_channels,figsize=(2 * num_channels, 2 * n_filters))
            ax = np.atleast_1d(ax)
            for i in range(n_filters):
                for j in range(num_channels):
                    ax[i][j].imshow(x[i,j], cmap="gray")
                    ax[i,j].set_xticks([])
                    ax[i,j].set_yticks([])
                    
                    if (i==0) and (j==0):
                        ax[i,j].set_title(note)         

        else:
            # activations map
            num_channels = x.shape[0]
            height=x.shape[1]/10
            width=x.shape[2]/10
            
            fig, ax = plt.subplots(1,num_channels,figsize=(2*width*num_channels, 2*height))
            # Ensure axes is subscriptable as a 1D array
            ax = np.atleast_1d(ax)
            
            for j in range(num_channels):
                ax[j].imshow(x[j], cmap="gray")
                if j==0:
                    ax[j].set_title(note)
                ax[j].set_xticks([])
                ax[j].set_yticks([])

        fig.tight_layout()





    def _visualize_one_output(self, images_batch, labels, idx=0):
        self._visualize_tensor(images_batch[idx], note=f"Original\nclass={labels[idx]}")
        for layer in self.layers_to_hook:
            # 2. Plot All Activations across the bottom row: ax[1, :]
            self._visualize_tensor(self.activations[layer][idx], note=layer)

    def _visualize_output(self,images_batch,labels,n_images):
        for i in range(n_images):
            self._visualize_tensor(images_batch[i], note=f"Original\nclass={labels[i]}")
            for layer in self.layers_to_hook:
                    #print(layer)
                    # 2. Plot All Activations across the bottom row: ax[1, :]
                    self._visualize_tensor(self.activations[layer][i],note=layer)
