# Custom Data Parallel for PyTorch

## Usage
```bash
python test.py -h

usage: test.py [-h] [--model MODEL] [--epoch EPOCH] [--batch-size BATCH_SIZE]

optional arguments:
  -h, --help            show this help message and exit
  --model MODEL         [single, future, threading, nn (nn.DataParallel)]
  --epoch EPOCH         default: 1
  --batch-size BATCH_SIZE   default: 1024

```
## Environment

| | |
|---|---|
| GPU | 8 GPUs (TESLA_P40) |
| CPU | 48 CPUs |
| Dataset | CIFAR10 |
| Model | ResNet50 |

## Results

| model | epoch | elasped time | accuracy |
|---|---|---|---|
| single | 10 | 276.26(s) | 99.04% |
| future | 10 | 132.00(s) | 91.73% |
| threading | 10 | 164.11(s) | 91.91% |
| nn.DataParallel | 10 | 132.92(s) | 91.72% |
