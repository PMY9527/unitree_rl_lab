# 代码变更说明

## 概述
本次变更主要实现了基于残差学习(Residual Learning)的CMG运动跟踪训练系统，并优化了训练参数和奖励函数。

---

## 1. 路径配置 (`unitree.py`)

**文件位置**: `source/unitree_rl_lab/unitree_rl_lab/assets/robots/unitree.py`

### 变更内容:
- 更新了Unitree模型和ROS包的路径配置
  - `UNITREE_MODEL_DIR`: `"path/to/unitree_model"` → `"/root/gpufree-data/unitree_ws/unitree_model"`
  - `UNITREE_ROS_DIR`: `"path/to/unitree_ros"` → `"/root/gpufree-data/unitree_ws/unitree_ros"`

---

## 2. PPO训练配置 (`rsl_rl_ppo_residual_cfg.py`)

**文件位置**: `source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/agents/rsl_rl_ppo_residual_cfg.py`

### 变更内容:
1. **训练迭代次数**: `50000` → `10000`
2. **经验归一化**: `False` → `True` (启用观测归一化)
3. **初始噪声标准差**: `1.0` → `0.2` (降低初始探索噪声)
4. **网络结构**: 保持为 `[512, 256, 128]` (注释中包含了 `[256]` 的备选方案)
5. **熵系数**: `0.01` → `0.005` (降低探索倾向，促进策略收敛)

### 设计思路:
- 降低训练迭代次数，加快实验迭代速度
- 启用经验归一化提高训练稳定性
- 调整超参数以适应残差学习任务

---

## 3. 奖励函数 (`rewards.py`)

**文件位置**: `source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/mdp/rewards.py`

### 变更内容:

#### 3.1 CMG跟踪奖励函数的安全检查
在以下两个函数中添加了空值检查:
- `joint_pos_from_cmg_l2()`: CMG关节位置跟踪奖励
- `joint_vel_from_cmg_l2()`: CMG关节速度跟踪奖励

**添加的逻辑**:
```python
if ref_motion is None:
    # CMG motion not set (e.g., during play mode) - return zero reward
    return torch.zeros(env.num_envs, device=env.device)
```

**原因**: 在play模式下，CMG motion可能未设置，需要避免访问空引用导致的错误。

#### 3.2 新增动作平滑度奖励函数
新增函数 `action_smoothness_l2()`:
- 计算动作的二阶导数(抖动): `jerk = a_t - 2*a_{t-1} + a_{t-2}`
- 返回: `||jerk||^2`
- 用途: 惩罚动作的急剧变化，使机器人运动更加平滑自然

**实现细节**:
- 使用 `env.extras["_action_smoothness_prev_prev"]` 存储 `a_{t-2}`
- 首次调用时返回零奖励(因为缺少历史数据)

---

## 4. 环境配置 (`RuN_env_cfg.py`)

**文件位置**: `source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/robots/g1/29dof/RuN_env_cfg.py`

### 变更内容:

#### 4.1 地面摩擦力参数
- **静摩擦系数**: `(0.3, 1.0)` → `(0.6, 1.0)`
- **动摩擦系数**: `(0.3, 1.0)` → `(0.4, 0.8)`

**原因**: 提高最小摩擦力，使训练环境更接近真实地面条件。

#### 4.2 速度指令范围
**训练模式** (`CommandsCfg`):
- **前向速度**: `(-0.1, 0.5)` → `(0.0, 3.0)` m/s
- **侧向速度**: `(-0.1, 0.1)` → `(-0.3, 0.3)` m/s
- **角速度**: `(-0.1, 0.1)` → `(-0.5, 0.5)` rad/s

**限制范围**:
- **前向速度**: `(-0.5, 3.0)` → `(0.0, 3.5)` m/s
- **侧向速度**: 保持 `(-0.3, 0.3)` m/s
- **角速度**: `(-0.2, 0.2)` → `(-0.5, 0.5)` rad/s

**原因**: 扩大训练速度范围，提高策略泛化能力，支持更高速运动。

#### 4.3 动作配置 (残差学习关键变更)
```python
# 旧配置
scale=0.25, use_default_offset=True

# 新配置
scale=1.0, use_default_offset=False
```

**原因**:
- 残差学习中，动作已经是绝对关节位置参考值
- 不需要缩放和默认偏移
- `scale=1.0` 保持原始输出

#### 4.4 奖励权重调整

| 奖励项 | 旧权重 | 新权重 | 变化说明 |
|--------|--------|--------|----------|
| `flat_orientation_l2` | -2.0 | -5.0 | 增强姿态控制约束 |
| `action_rate` | -0.04 | -0.02 | 降低动作变化率惩罚 |
| `action_smoothness` | - | -0.04 | **新增**平滑度惩罚 |
| `joint_deviation_arms` | -0.2 | -0.1 | 降低手臂偏差惩罚 |

**删除的参数**:
- `track_lin_vel_xy`: 删除了 `std=math.sqrt(0.25)` 参数
- `track_ang_vel_z`: 删除了 `std=math.sqrt(0.25)` 参数

#### 4.5 Play模式配置 (`RuNPlayEnvCfg`)
新增配置:
- **环境数量**: 32个并行环境
- **环境间距**: 2.5米
- **地形大小**: 2行 × 5列
- **速度指令**: 仅前向运动
  - `lin_vel_x`: `(0.0, 1.0)` m/s
  - `lin_vel_y`: `(0.0, 0.0)` m/s (禁用侧向)
  - `ang_vel_z`: `(0.0, 0.0)` rad/s (禁用旋转)

---

## 5. 训练脚本 (`train.py`)

**文件位置**: `scripts/rsl_rl/train.py`

### 变更内容:
```python
# 旧代码
from rsl_rl.runners import OnPolicyRunner
runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)

# 新代码
from rsl_rl.runners import OnPolicyRunner, OnPolicyRunnerResidual
runner = OnPolicyRunnerResidual(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
```

**说明**: 切换到 `OnPolicyRunnerResidual`，这是一个专门用于残差学习的训练器。

---

## 6. 推理脚本 (`play.py`)

**文件位置**: `scripts/rsl_rl/play.py`

### 变更内容:

#### 6.1 添加 `OnPolicyRunnerResidual` 支持
```python
elif agent_cfg.class_name == "OnPolicyRunnerResidual":
    from rsl_rl.runners import OnPolicyRunnerResidual
    runner = OnPolicyRunnerResidual(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
```

#### 6.2 观测数据处理重构
**旧代码**:
```python
obs = env.get_observations()
actions = policy(obs)
obs, _, _, _ = env.step(actions)
```

**新代码**:
```python
# 处理元组返回: (policy_tensor, {"observations": obs_dict})
if isinstance(obs, tuple):
    obs_tensor, obs_extras = obs
    obs = TensorDict(obs_extras.get("observations", {"policy": obs_tensor}), batch_size=[env.num_envs])

# 获取原始机器人数据用于CMG
robot = env.unwrapped.scene["robot"]
robot_data = (robot.data.joint_pos, robot.data.joint_vel)

# 传递机器人数据到策略
actions = policy(obs, robot_data=robot_data)

# 重建TensorDict
obs_tensor, _, _, extras = env.step(actions)
if "observations" in extras:
    obs = TensorDict(extras["observations"], batch_size=[env.num_envs])
else:
    obs = TensorDict({"policy": obs_tensor}, batch_size=[env.num_envs])
```

**关键改进**:
1. 正确处理IsaacLab wrapper返回的元组格式
2. 直接获取原始机器人关节状态(避免观测缩放问题)
3. 将原始数据传递给策略，用于CMG参考跟踪
4. 使用TensorDict统一管理多模态观测数据

---

## 7. OnPolicyRunnerResidual

**说明**: `OnPolicyRunnerResidual` 是从 `rsl_rl.runners` 导入的外部类，不是本仓库的一部分。

### 使用场景:
- 专门设计用于残差学习训练
- 在训练和推理时都需要CMG参考运动数据
- 策略输出残差动作，叠加到参考轨迹上

### 相关文件:
- 配置文件: `rsl_rl_ppo_residual_cfg.py` (指定 `class_name = "OnPolicyRunnerResidual"`)
- 训练脚本: `train.py` (实例化runner)
- 推理脚本: `play.py` (根据class_name加载对应runner)

---

## 总结

### 核心改进:
1. **残差学习架构**: 完整实现了CMG运动跟踪的残差学习流程
2. **训练稳定性**: 启用经验归一化，调整超参数
3. **运动质量**: 新增动作平滑度奖励，调整摩擦力参数
4. **泛化能力**: 扩大速度指令范围，支持高速运动
5. **代码健壮性**: 添加空值检查，改进观测数据处理

### 技术特点:
- 残差学习允许策略在CMG参考轨迹基础上微调
- 动作输出直接作为绝对关节位置(scale=1.0)
- 在推理时直接使用原始机器人状态，避免观测缩放带来的误差

### 下一步建议:
- 监控 `action_smoothness` 奖励的影响
- 评估扩大速度范围后的策略性能
- 验证摩擦力参数在真实机器人上的效果
