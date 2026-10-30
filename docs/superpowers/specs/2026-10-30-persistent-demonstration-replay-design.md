# 持续模仿与独立示范池设计

## 目的

防止TD3-BC在示范退出ReplayBuffer、模仿权重降为零后发生策略退化，同时保持标准TD3/DDPG完全不接触MapGuide示范。

## 设计

- TD3-BC/DDPG-BC的模仿权重从1.0线性衰减，在0.25处保持；标准算法权重恒为0。
- ReplayBuffer包含互不覆盖的在线池和示范池。MapGuide整回合转换只进入示范池，普通Actor转换只进入在线池。
- 每次256条训练batch优先抽取192条在线经验和64条示范经验；任一池为空时由另一池补足。
- Critic使用完整混合batch，Actor的BC项仅使用示范掩码样本。
- 配置和TensorBoard记录BC下限、示范比例、在线池大小和示范池大小。
- 新正式结果写入 `journal_results/formal-v5`；A变体下同时启动Low/Medium/High三组TD3-BC 3000轮，以及Low一组DDPG-BC 3000轮。

