# Paper Reading List

## 2022-02

### HeteroFL: Computation and Communication Efficient Federated Learning for Heterogeneous Clients [[pdf](https://arxiv.org/pdf/2010.01264.pdf)]

> Heterogeneous

Introduce a framework called HeteroFL to train heterogeneous local models with varying computation complexities and still produce a single global inference model.

The first work to allow local models to have different architectures from the global model.

![](C:\Users\scott\Documents\paper-reading\assets\2022-02-09-20-01-10-image.png)

### How To Backdoor Federated Learning [[pdf](https://arxiv.org/pdf/1807.00459.pdf)]

> Security

Federated learning is generically vulnerable to model poisoning

- it is impossible to ensure that none of millions of participants are malicious

- neither defenses against data poisoning, nor anomaly detection can be used during federated learning
  
  because they require access to the training data or model updates.

![](C:\Users\scott\Documents\paper-reading\assets\2022-02-09-19-59-41-image.png)

### Federated Learning for Keyword Spotting [[pdf](https://arxiv.org/pdf/1810.05512.pdf)]

> application for wake word detectors

This paper introduce Federated Learning method with adaptive averaging strategy into wake word detection task.