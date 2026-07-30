# CarryBox Sim2Sim 诊断报告

## 当前结果

共享部署链已经不再是两个 actor 同时失败的原因。
2026-07-29 进行的严格 20 秒双进程 UDP 验证结果如下：

| 模型配置 | 结果 | Policy / Physics 频率 | 推理耗时 p99 | 最大 XY 倾斜量 | 最大关节限位穿透 | 首次失败 |
|---|---:|---:|---:|---:|---:|---|
| `official_carrybox_65000` | 通过 | 50.01 / 199.76 Hz | 0.63 ms | 0.358 | 0.0116 rad | 无 |
| `model_55500` | 失败 | 终止前 49.94 / 199.85 Hz | 0.48 ms | 0.620 | 0.0052 rad | 约 11.4 秒触发训练环境的 roll 终止条件 |
| `model_73500` | 失败 | 50.03 / 199.91 Hz | 0.49 ms | 0.733 | 0.0013 rad | 约 11.1 秒双侧 hip-yaw link 接触地面 |

两个本地训练的 policy 几乎在相同任务时刻失败，但终止方式不同。
`model_55500` 的侧倾超过 Isaac Gym 的终止阈值
（`roll=-0.514 rad`）；`model_73500` 的数值始终有限，但逐渐收敛到
极低蹲姿，最终双侧 hip-yaw link 接触地面。这些属于任务/轨迹一致性
失败，而不是通信、manifest、关节映射、action decimation 或推理频率
问题。

失败时，两个 policy 都没有将箱子抬离地面。`model_55500` 只将箱子
移动了约 `[+0.058, -0.157, 0.000] m`，`model_73500` 将箱子移动了约
`[+0.370, -0.280, 0.000] m`。在相同部署链下，官方 actor 能保持直立
运行 20 秒，因此可以排除“某个共享部署错误导致所有 actor 都不稳定”
这一可能。

## 训练证据

TensorBoard 历史数据表明，继续优化提高了总 reward，但没有改善实际
测量到的搬运交互质量：

- Mean reward 从迭代 55,500 附近的约 `37.4` 上升到迭代 73,500
  附近的 `49.2`。
- Confirmed-carry ratio 从约 `0.0525` 下降到 `0.0413`。
- Both-hand contact ratio 从约 `0.1235` 下降到 `0.1055`。
- Lifted bimanual contact ratio 从约 `0.469` 下降到 `0.400`。
- 关节限位惩罚的绝对值从约 `0.106` 恶化到 `0.177`。

当前训练配置还关闭了长距离搬运进度奖励、搬运稳定性奖励和成功终止
奖励。Hybrid reset 有 80% 的概率从参考动作初始化，而确定性的 MuJoCo
测试需要 policy 从完整的接近箱子和抓取阶段开始执行。因此，更高的
PPO return 并不能证明 policy 具有更稳定的端到端搬运能力。

当前 MuJoCo 场景也没有复现训练任务的全部细节：Isaac Gym 使用按任务
阶段划分的动作状态、随机箱体属性以及起点/终点平台。在将全部失败
归因于物理引擎迁移之前，必须使用确定性的 Isaac Gym 分阶段快照分别
隔离这些任务和接触差异。

## 已确认并修复的部署问题

- Actor observation 现在使用训练时的 `pelvis` policy frame 构造，
  不再错误使用 `torso_link`。
- 只有当 robot state 和 task state 的源 sequence 完全一致时，才将
  两者配对。
- 重复 sequence packet 会被跳过，并且不会清空六帧 observation
  history；只有 sequence 真正回退时才会重置历史。
- UDP 协议版本 2 会拒绝使用旧 torso 坐标语义的数据包。
- MuJoCo server 使用 `0.01 m` 碰撞 margin，并为关节限位显式设置
  `solref`/`solimp` 参数。
- Position target 只会在每四个 physics step 的 policy boundary 上
  更新。
- Sim-parity 的 episode failure 已与硬件 fail-closed 行为分离。
  模拟器会暂停并锁存 Isaac 风格的终止状态，不再切换到零 Kp fallback
  后任由机器人在重力作用下倒地。
- 200 Hz deadline scheduler 不再累计每个 tick 的 sleep 误差。
- 每个模型 profile 都使用独立的 ONNX/manifest 路径，并强制校验
  SHA256。
- Validator 要求整个有效阶段连续通过，并检查倒地、异常地面接触、
  关节限位、sequence、控制 decimation 和延迟。

## 仍需在 Isaac Gym 实验室主机上完成的验证

需要使用一个故意设置为非零 waist 姿态的状态，运行 golden snapshot
导出和比较工具。Isaac Gym 原始状态与部署侧重建 observation 的误差
必须小于 `1e-5`。这是当前针对实际训练环境尚未完成的权威一致性检查。

任务级一致性验证还需要分别导出接近箱子（`loco`）、抓取（pickup）、
搬运（carry）和放下（putdown）四个阶段的确定性 Isaac Gym 初始状态。
在将这些状态及其 Isaac Gym baseline 重放到 MuJoCo 之前，当前 validator
只能证明部署/控制接口工作正常，并证明官方 actor 能够保持直立；它
不能证明四阶段搬运成功率已经达到 80%。

不得通过缩小 action scale、降低 Kp 或压缩 actor 输出来掩盖失败阶段。
应将首次产生差异的 observation、action 和 contact trace 与对应的
Isaac Gym golden trace 逐项比较。
