import numpy as np

def AccuarcyCompute(pred, label):
    pred = pred.cpu().data.numpy()
    label = label.cpu().data.numpy()
    test_np = np.argmax(pred, 1)
    count = 0
    for i in range(len(test_np)):
        if test_np[i] == label[i]:
            count += 1
    return np.sum(count), len(test_np)
