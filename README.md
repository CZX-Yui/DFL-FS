# Main framework

![framework](https://github.com/CZX-Yui/DFL-FS/assets/59764728/97876d44-7dfb-4dbb-b2bd-b1004960c8f7)

We decouple the model into a feature extractor F and a classifier H within a two-stage training framework. In the first stage (1) the clients train the local model and calculate feature statistics \mu_z. (2) The server then estimates the client’s coverage distribution through clustering masked feature statistics and filters class-balanced clients for model aggregation. In the second stage, (3) the clients use the frozen global feature extractor to calculate local feature statistics, and upload them to the server, (4) the server then calculates the global feature statistics to regenerate samples from Gaussian distribution by resampling or weighted covariance to retrain the classifier.
