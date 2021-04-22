This folder contains source codes for the report

**The Procurement-VCG Mechanism for Encouraging Data Contribution in Federated Learning**

In this report, we propose to learn the optimal PVCG adjustment payments through automated mechanism design. Purposes of our experimental evaluation include:

The detailed experimental setup is described in the report

***************

**Our programs have been tested with Python 3.6.8 and Tensorflow 1.14.0 in Linux**

Files in this folder include:

* model: save trained parameters of the neural network.
* model_initialize_do_not_delete: save randomly generated initialization parameters (to guarantee repeatability of our experiment)
* train.py: implementation of Algorithm 1 in the technical appendix
* construct_graph.py: construct the composite neural network in Figure 1
* compute_acceptance.py: calculate the acceptance ratio in Step 1 of the procurement auction
* compute_social_surplus.py: calculate the optimal social surplus
* compute_tau.py: calculate the VCG payment in Eq. (2)
* plt_loss.py: plot the loss curve
* plt_payment.py: plot the payment surface in Figure 1
* loss.txt: save training losses
