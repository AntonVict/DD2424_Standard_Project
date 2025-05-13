# Results

## Stage 1

### Binary Classification

[Telemetry] Epoch 3 | Train Loss: 0.0187 | Train Acc: 0.9951
[Telemetry] Epoch 3 | Val Loss: 0.0330 | Val Acc: 0.9877

## Stage 2

### Multi-class Classification

#### Strategy 1

[Telemetry] Epoch 3 | Train Loss: 1.8400 | Train Acc: 0.8136
[Telemetry] Epoch 3 | Val Loss: 1.7968 | Val Acc: 0.7261

#### Strategy 2

[Telemetry] Epoch 3 | Train Loss: 2.0806 | Train Acc: 0.6845
[Telemetry] Gradual: Unfroze layer 2
[Telemetry] Epoch 3 | Val Loss: 1.7762 | Val Acc: 0.7604

#### Conclusion

Strategy 2 is better than Strategy 1 when epochs = 3.
