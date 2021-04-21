This folder contains codes for preliminary experimental evaluation of the PVCG sharing rule in the following paper:

**Optimal Procurement Auction for Cooperative Production of Virtual Goods**

In this paper, we propose to learn the optimal PVCG adjustment payments through automated mechanism design. Purposes of our experimental evaluation include:

* To illustrate the effectiveness of the automated mechanism design method (Algorithm 1 in the paper's technical appendix)
* To show the loss function can be minimized to 0, demonstrating the soundness of Theorem 3, which has been proved in the paper

Other purposes, such as implementing the procurement auction in real scenarios and comparing the PVCG sharing rule with other payoff-sharing rules, are beyond the scope of the aforementioned paper

In this experiment, we learn the PVCG payments for a hypothetical scenario with the following individual valuation function and individual cost function:

$$v(x)=\theta_i\sqrt{n(\sum_{k-0}^{n-1}x_k)} \quad \textrm{and} \quad c(x_i,\gamma_i)=\gamma_i x_i , \quad i\in N$$

The detailed experimental setup is provided in the paper's technical appendix

***************

**Our programs have been tested with Python 3.6.8 and Tensorflow 1.14.0 in Linux**

Files in this folder include:

* model: save trained parameters of the neural network.
* model_initialize: save randomly generated initialization parameters (to guarantee repeatability of our experiment)
* train.py: implementation of Algorithm 1 in the technical appendix
* construct_graph.py: construct the composite neural network in Figure 1
* compute_acceptance.py: calculate the acceptance ratio in Step 2 of the procurement auction
* compute_social_surplus.py: calculate the optimal social surplus
* compute_tau.py: calculate the VCG payment in Eq. (7)
* plt_loss.py: plot the loss curve in Figure 2
* plt_payment.py: plot the payment surface in Figure 2
* loss.txt: save training losses
