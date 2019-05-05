## Speaker Recognition -- One-shot Learning
----------------------------
#### A few loss functions like triplet loss, l2-softmax loss and siamese loss were realized.What you need to do is to find them out.

1. Train data structure:  
```
.
├── train_dir
|   └── ID-1
|       ├─── wav-1
|       ├─── wav-2
|       *
|       *
|   └── ID-2
|       ├─── wav-1
|       ├─── wav-2
|       *
|       *
|   └── *
|   └── *
```
2. Train command:   
`usage: speaker recognition [-h] [--file_dir FILE_DIR]
                           [--model_path MODEL_PATH]
                           [-s OUTPUT_SHAPE OUTPUT_SHAPE]
                           [--hidden_size HIDDEN_SIZE] [-e EPOCHS_TO_TRAIN]
                           [-b BATCH_SIZE] [-sr SAMPLE_RATE] [-c CLASS_NUM]
                           [-pc PROCESS_CLASS] [-mt MODEL_TYPE] [-n NET_DEPTH]
                           [-fl FEATURE_LENGTH] [-lc LAMBDA_C] [-l2 L2_LAMBDA]
                           [--continue_training]`
     
3. Test:  split some files out, choose model, load type and set other parameters then run model_test.py.

