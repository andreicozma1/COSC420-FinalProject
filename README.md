# CS420 - Final Project 

 ## Abstract  
 
 The current state of Spiking Neural Networks (SNNs) has shown great potential in outperforming traditional Artificial Neural Networks (ANNs) in many ways. Biologically- inspired neuron models have also shown to offer improved learning capabilities, and neuromorphic implementations promise improved energy efficiency, scalability, and performance. This paper seeks to evaluate the performance and behaviors of spiking networks as compared to the classic networks for the OpenAI Gym CartPole environment, which is a classic real- time control problem. The experimental approach contributes in two main areas through the use of a stochastic policy gradient reinforcement learning method, as well as a recurrent spiking network architecture. The first experiment compared the performance between the two in the CartPole environment, with the SNNs significantly outperforming the ANNs in terms of average reward, but at the cost of increased variance in the scores between different training trials. The second experiment explores the effect of hyper-parameter tuning on the performance of the spiking network, which outperforms the classic network to an even greater degree and brings some improvements to the variability. Lastly, this work provides insights into potential ways to further improve the learning and generalization capabilities of SNNs in reinforcement learning problems in the future.

 ## Usage Example:

```
python3 main.py --environment {env} --policy {policy} --episodes {episodes} --lr {lr} --gamma {gamma} --h1 {h1} --h2 {h2} --dropout {dropout} --sqlen {sqlen} --scale {scale} --quiet True --seed -1 --subdir {subdir}
```