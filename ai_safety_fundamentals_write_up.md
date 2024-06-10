# Investigating Weak to strong Generalisation in open source models

## Introduction

For my AI safety fundamentals capstone project, i have decided to investigate wether the results from OpenAI's 2023 paper '[Weak-to-Strong Generalization: Eliciting Strong Capabilities With Weak Supervision](https://cdn.openai.com/papers/weak-to-strong-generalization.pdf)' can be replicated using open source models. I believe that weak to strong generalisation is an important area of study in ai safety, as I feel that finding ways to improve the ability of strong models to learn from weak supervisors is essential in improving the safety of superhuman AI models, as these models will likely need to be alligned by supervisors that are weaker than them (humans). For my experiment, I have chosen to use googles bert and Gemma-2b models as my weak and strong models respectively. I believe that my experiment will contribute towards AI safety as I believe that it is important to gather more datapoints, wether that is with different model combinations, or different tasks, in order to gain more confidence in the conclusions of OpenAIs original paper.

## Experimental setup

For my experiment, I decided to solely focus on the NLP task detailed in the paper. This is because OpenAI has only open sourced their code for this task, and I did not have time to implement the other two tasks. In order to train both models I have used the same hyperparamers that were used in the original experiment. The only way that my experiment differs is that I decided to use parameter efficiant fine tuning, specifically [LoRA](https://arxiv.org/abs/2106.09685) (Low Rank Adaptation). I decided to use LoRA to train my models, as i only had access to an Nvidia T4 GPU via google collab, and hence did not have the resources to train Gemma using conventional model fine tuning methods.

## Results

Unfortunately, I was unable to collect any meaningfull results during this experiment, as I had issues with exploding gradients when training both models, which I was unable to get to the bottom of during the project sprint.