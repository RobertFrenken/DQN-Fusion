    def distill_autoencoder_simplified(self, train_loader: DataLoader, epochs: int = 20, 
                                     alpha: float = 0.7, temperature: float = 5.0):
        """
        Simplified autoencoder distillation focusing on latent space and reconstruction.
        
        This method provides 3-4x speedup by focusing only on the most critical
        knowledge transfer components: VGAE latent space and reconstruction quality.
        
        Args:
            train_loader: DataLoader with normal graphs for training
            epochs: Number of training epochs
            alpha: Weight for distillation loss vs. task loss
            temperature: Temperature for knowledge distillation
        """
        print(f"\n=== Simplified Autoencoder Distillation ===")
        print(f"Epochs: {epochs}, Alpha: {alpha}, Temperature: {temperature}")
        print("Focus: VGAE latent space + reconstruction (3-4x speedup)")
        
        self.student_autoencoder.train()
        
        # Optimizer setup
        optimizer = torch.optim.Adam(
            self.student_autoencoder.parameters(), 
            lr=4e-3, 
            weight_decay=1e-5
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.7, patience=3, verbose=True
        )
        
        # Handle dimension mismatch between teacher and student
        projection_layer = self._setup_projection_layer()
        proj_optimizer = None
        if projection_layer is not None:
            proj_optimizer = torch.optim.Adam(projection_layer.parameters(), lr=1e-3)
        
        # Mixed precision setup for CUDA
        scaler = torch.cuda.amp.GradScaler() if self.is_cuda else None
        
        # Training loop
        best_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            num_batches = 0
            
            for batch_idx, batch in enumerate(train_loader):
                batch = batch.to(self.device, non_blocking=True)
                
                # Compute loss with mixed precision if available
                if self.is_cuda:
                    with torch.amp.autocast('cuda', dtype=torch.float16):
                        total_loss = self._compute_simplified_autoencoder_loss(
                            batch, projection_layer, alpha, temperature)
                else:
                    total_loss = self._compute_simplified_autoencoder_loss(
                        batch, projection_layer, alpha, temperature)
                
                # Backward pass
                self._perform_backward_pass(total_loss, optimizer, proj_optimizer, scaler, 
                                          self.student_autoencoder, projection_layer)
                
                epoch_loss += total_loss.item()
                num_batches += 1
                
                # Cleanup and memory management
                del batch, total_loss
                if batch_idx % 20 == 0:
                    cleanup_memory()
            
            # Epoch statistics
            avg_loss = epoch_loss / max(num_batches, 1)
            
            # Learning rate scheduling
            scheduler.step(avg_loss)
            
            # Early stopping
            if avg_loss < best_loss * 0.999:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Progress reporting
            if (epoch + 1) % 5 == 0:
                print(f"Autoencoder Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")
                log_memory_usage(f"AE Epoch {epoch+1}")
            
            # Early stopping check
            if patience_counter >= 8:
                print(f"Early stopping: No improvement for {patience_counter} epochs")
                break
            
            # Inter-epoch cleanup
            cleanup_memory()
        
        # Set anomaly detection threshold
        self._set_student_threshold(train_loader)
        print(f"✓ Student threshold set: {self.threshold:.4f}")

        def _setup_projection_layer(self) -> Optional[nn.Module]:
            """Setup projection layer if teacher and student have different latent dimensions."""
            teacher_latent_dim = getattr(self.teacher_autoencoder, 'latent_dim', 32)
            student_latent_dim = getattr(self.student_autoencoder, 'latent_dim', 16)
            
            if teacher_latent_dim != student_latent_dim:
                projection_layer = nn.Linear(student_latent_dim, teacher_latent_dim).to(self.device)
                print(f"✓ Projection layer added: {student_latent_dim} → {teacher_latent_dim}")
                return projection_layer
            return None