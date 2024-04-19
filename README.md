# OffPolicy-PPO
Unofficial implementation of [Off-policy Proximal Policy Optimization](https://ojs.aaai.org/index.php/AAAI/article/view/26099) in PyTorch

# Update

- 2024-04-19
    - Add prioritized replay buffer by referring to [Actor Prioritized Experience Replay](https://www.jair.org/index.php/jair/article/view/14819)


# Train
Find or make a config file and run the following command.
```
python main.py --config=configs/Ant-v4.yaml 
               --exp_name=Ant-v4_v1 
               --train
```

# Result

## V-trace version
![](https://github.com/Ladun/OffPolicy-PPO/blob/master/plots/plot.jpg)

# GAE version
![](https://github.com/Ladun/OffPolicy-PPO/blob/master/plots/gae_ver.jpg)