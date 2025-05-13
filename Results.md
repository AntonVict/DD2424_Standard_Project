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

## Fine-tuning with Imbalanced Classes

**Training class distribution (label: count):**
[(0, 20), (1, 100), (2, 100), (3, 100), (4, 100), (5, 20), (6, 20), (7, 19), (8, 100), (9, 20), (10, 100), (11, 18), (12, 96), (13, 100), (14, 100), (15, 100), (16, 100), (17, 100), (18, 100), (19, 100), (20, 20), (21, 100), (22, 96), (23, 20), (24, 100), (25, 100), (26, 20), (27, 20), (28, 100), (29, 100), (30, 100), (31, 100), (32, 19), (33, 20), (34, 100), (35, 100), (36, 100)]

**Number of training samples:** 2728
**Number of test samples:** 3669
**Batch size:** 32
**Device:** mps
**Cat keep fraction:** 0.2

**Validation accuracy after 3 epochs:** 0.8580

**Per-class accuracy:**

  Class  1: 0.7143 (70/98)
  Class  2: 0.9200 (92/100)
  Class  3: 0.5900 (59/100)
  Class  4: 0.8700 (87/100)
  Class  5: 0.9000 (90/100)
  Class  6: 0.9400 (94/100)
  Class  7: 0.8100 (81/100)
  Class  8: 0.7727 (68/88)
  Class  9: 0.9192 (91/99)
  Class 10: 0.8000 (80/100)
  Class 11: 0.8400 (84/100)
  Class 12: 0.7216 (70/97)
  Class 13: 0.8400 (84/100)
  Class 14: 0.9400 (94/100)
  Class 15: 0.9800 (98/100)
  Class 16: 0.9800 (98/100)
  Class 17: 0.9600 (96/100)
  Class 18: 0.9700 (97/100)
  Class 19: 1.0000 (99/99)
  Class 20: 0.9700 (97/100)
  Class 21: 0.5500 (55/100)
  Class 22: 0.9300 (93/100)
  Class 23: 1.0000 (100/100)
  Class 24: 0.7000 (70/100)
  Class 25: 0.9800 (98/100)
  Class 26: 0.9500 (95/100)
  Class 27: 0.7500 (75/100)
  Class 28: 0.6800 (68/100)
  Class 29: 0.9800 (98/100)
  Class 30: 0.9900 (99/100)
  Class 31: 0.9899 (98/99)
  Class 32: 0.9200 (92/100)
  Class 33: 0.6300 (63/100)
  Class 34: 0.8800 (88/100)
  Class 35: 0.5618 (50/89)
  Class 36: 0.8800 (88/100)
  Class 37: 0.8900 (89/100)


#### Conclusion

The model performs well on majority classes but struggles on underrepresented (minority) classes, highlighting the impact of class imbalance.

