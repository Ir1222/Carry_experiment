# Humanoid object interaction 中 contact / force reward 的论文-代码审计

审计日期：2026-06-28  
当前仓库：`PhysHSI`  
重点：HAIC、AdaptManip、Sim-to-Real Learning for Humanoid Box Loco-Manipulation、HDMI，以及当前 CarryBox 环境的可落地设计。

## 1. 结论先行

四篇工作的 contact/force 设计可以归为三类：

1. **目标接触点 + 最小接触力门槛**：HDMI、HAIC。先让 end-effector 到达物体上的语义接触点，再要求接触力至少达到阈值；超过阈值后奖励饱和，而不是继续增大。
2. **稳定抓持 + 相对运动/滑移约束**：AdaptManip。在位置、物体稳定之外，用接触力和手-物体相对速度约束抓持质量。
3. **阶段化接触 + 小力/低冲击正则**：2023 Digit box 工作。先规定何时接触、何时起箱；接触本身由离散 indicator 控制，同时惩罚手力、桌-箱力和箱体加速度。

对当前仓库，最重要的判断是：

- 现有 critic 已经有 box 速度、双手净接触力、box 净接触力和双手 contact flag，具备做 asymmetric contact-aware critic 的基础。
- 但当前 task reward 没有真正使用这些量；`hand_contact` 权重为 0，`carryup_task` 只奖励双手均值靠近箱体中心和箱体高度。
- 当前 `net_contact_force_tensor` 是**刚体净力**，不是 hand-box **pair-specific contact force**。它可能混入手-地面、手-机器人或 box-table 接触；box 两侧相反方向的夹持力还可能在净和中互相抵消。因此不能直接把 `box_net_contact_force` 当抓持力真值。
- 最稳妥的起点是采用 HAIC/HDMI 的“接触点距离 × 最小法向力”主奖励，再补 AdaptManip 启发的双手对称与切向滑移惩罚，以及 Digit 工作的冲击/过力惩罚。

## 2. 横向对比

| 工作 | 接触时序 | 位置项 | 力项 | 滑移/稳定项 | force 是否进入 critic | 官方代码状态 |
|---|---|---|---|---|---|---|
| HDMI | reference contact label `c_{t,i}` | EEF 到目标接触点 | 低于 `F_thres` 扣分，超过后饱和 | 主要依赖 object pose tracking 和 lost-contact termination | 论文/当前开源配置中不是显式 critic force channel | 已开源 |
| HAIC | `I_{o,e}`，来自 reference contact mask | 有 `epsilon_tol` 容差 | 低于 `F_thr` 扣分 | 多对象平均、object tracking、foot/impact constraints | critic 有 privileged object state、applied force/torque 等 | 已开源，但多对象分支存在实现问题 |
| AdaptManip | 抓取/运输连续控制 | robot-box yaw、hand/root error | 接触力质量项 | root-box 相对速度、双手抓持、slip avoidance | 明确包含 `f_hand`、`f_box` 和 box 速度 | 未发现官方代码链接 |
| Digit Box | `t_contact`、`t_lift` 两阶段 | 三关键点手轨迹 | 接触 indicator；惩罚 hand force、table force | box acceleration、lost-contact termination | 未明确采用 asymmetric force critic | 未发现官方代码 |

## 3. HDMI

论文：[PDF](https://arxiv.org/pdf/2509.16757)；项目页：[HDMI](https://hdmi-humanoid.github.io/)；代码：[LeCAR-Lab/HDMI](https://github.com/LeCAR-Lab/HDMI)。本次代码审计提交为 `32282f6dcf26cae70b814d585ceb12cc38aa1b60`。

### 3.1 论文公式

对 active end-effector `i`：

```text
R_contact,i = exp(-||p_eef,i - p_target,i||_2 / sigma_pos)
              * min(exp((||F_contact,i||_2 - F_thres) / sigma_frc), 1)

R_interaction = (1 / N_c) * sum_i R_contact,i * c_t,i
```

变量：

- `p_eef,i`：第 `i` 个 end-effector 的世界位置。
- `p_target,i`：物体上为该 end-effector 指定的目标接触点。
- `F_contact,i`：end-effector 与物体之间的接触力向量。
- `F_thres`：认为接触力“足够”的下限。
- `sigma_pos`：位置误差衰减尺度；越小越要求精准。
- `sigma_frc`：力不足的衰减尺度；越小越强硬。
- `c_t,i`：参考轨迹在时刻 `t` 是否要求该 end-effector 接触；1 为 active。
- `N_c`：需要接触的 end-effector 数量。

第二个因子可等价写成：

```text
exp(-max(0, F_thres - ||F_contact,i||) / sigma_frc)
```

关键含义：这是**最小力门槛奖励**，不是力跟踪，也不是过力惩罚。超过 `F_thres` 后该项恒为 1；论文所说的 “bounded” 是奖励封顶，不代表大力会被惩罚。

论文表中 `Contact Reward` 权重为 5.0；lost-contact termination 使用位置 0.2 m、力 1 N、持续 25 steps。

### 3.2 代码核对

实现位于 [`rewards.py`](https://github.com/LeCAR-Lab/HDMI/blob/32282f6dcf26cae70b814d585ceb12cc38aa1b60/active_adaptation/envs/mdp/commands/hdmi/rewards.py#L367-L492)：

- `eef_pos_error = relu(||p_eef-p_target|| - pos_tolerance)`。
- `contact_frc = min(||F||-frc_thres, 0)`。
- `rew = exp(-eef_pos_error/pos_sigma) * exp(contact_frc/frc_sigma)`。
- 默认配置是 `pos_sigma=0.3`、`frc_sigma=40`、`frc_thres=10 N`、`gain=5`，见 [`hdmi-base.yaml`](https://github.com/LeCAR-Lab/HDMI/blob/32282f6dcf26cae70b814d585ceb12cc38aa1b60/cfg/task/base/hdmi-base.yaml#L220)。`gain=5` 与论文表中的 contact 权重 5 对应。
- `object_contact` 从 motion `.npz` 载入，作为 reference contact mask。
- 接触力来自 filtered contact sensor，按 EEF-object pair 汇总，然后旋转到 object frame。由于后续取范数，坐标变换不改变标量值。

差异：代码对 inactive contact 使用 `+ 1 - in_range`，因此非激活 EEF 贡献常数 1；论文公式则直接乘 `c_t,i`，inactive 项为 0。常数不改变该步的局部动作最优方向，但会改变 return/value 标度，属于论文-实现差异。

## 4. HAIC

论文：[PDF](https://arxiv.org/pdf/2602.11758)；项目页：[HAIC](https://haic-humanoid.github.io/)；代码：[ldt29/HAIC](https://github.com/ldt29/HAIC)。本次代码审计提交为 `e262500ee0e68e9d82f399aa18a4c10bf5b4ca1a`。

### 4.1 论文公式

```text
r_contact = (1/|O|) sum_{o in O} (1/|E_o|) sum_{e in E_o}
            I_o,e * r_pos^{o,e} * r_force^{o,e}

r_pos^{o,e} = exp(-max(0, ||p_e - p_tgt|| - epsilon_tol) / sigma_p)

r_force^{o,e} = exp(-max(0, F_thr - ||F_e||) / sigma_f)
```

变量：

- `O`：当前任务中的 active object 集合。
- `E_o`：与对象 `o` 交互的 end-effector 集合。
- `I_o,e`：reference motion 给出的二值接触 mask。
- `p_e`：end-effector 位置。
- `p_tgt`：该对象上的语义接触目标。
- `epsilon_tol`：位置容差带；误差小于它时位置项为 1。
- `sigma_p`：超出容差后的衰减尺度。
- `F_e`：对应 EEF-object 接触力。
- `F_thr`：最小接触力阈值。
- `sigma_f`：力不足的衰减尺度。

位置项与力项相乘很重要：只有“位置正确且接触力足够”才能拿满分，避免仅靠靠近或仅靠碰撞取巧。多对象先在各对象 EEF 内平均，再对对象平均，防止 EEF 数量多的对象支配总奖励。

论文的阈值消融测试了：`(epsilon_tol, F_thr)=(0.1 m,0 N)`、`(0.05 m,5 N)`、`(0 m,10 N)`；三者成功率均为 100%，但跟踪误差不同。Table X 给 Multiple Objects Contact 权重 1.0。

### 4.2 代码核对

核心类仍名为 [`eef_contact_exp`](https://github.com/ldt29/HAIC/blob/e262500ee0e68e9d82f399aa18a4c10bf5b4ca1a/active_adaptation/envs/mdp/commands/hdmi/rewards.py#L508-L625)：

- 默认 `pos_sigma=0.3`、`frc_sigma=40`、`frc_thres=10 N`、`gain=5`。
- `push_cart` 覆盖为 `pos_tolerance=0.1`、`frc_thres=0`，与论文第一组阈值相同。
- HAIC 将原 HDMI 的 `ref_object_contact` 改为 `ref_body_contact`，即按 body-object pair 激活；并支持第二对象。
- `body_contact` 与 `object_contact` 做逻辑与，mask 比 HDMI 的 object-level mask 更细。
- HAIC 去掉了 HDMI inactive contact 的常数 1，和论文公式更一致。

但当前公开提交的多对象路径有两个高置信问题：

1. [`command.py#L516-L529`](https://github.com/ldt29/HAIC/blob/e262500ee0e68e9d82f399aa18a4c10bf5b4ca1a/active_adaptation/envs/mdp/commands/hdmi/command.py#L516-L529) 把第二对象 contact buffer 的初始化缩进在 `if contact_eef_pos_offset_per_motion is not None` 下。公开配置没有设置该参数，因此 `object2` 任务随后访问 `contact2_target_pos_offset` 时可能直接触发 `AttributeError`。
2. [`rewards.py#L540-L545`](https://github.com/ldt29/HAIC/blob/e262500ee0e68e9d82f399aa18a4c10bf5b4ca1a/active_adaptation/envs/mdp/commands/hdmi/rewards.py#L540-L545) 对第二对象使用 `+=` 更新 `eef2_pos_error` 和 `eef2_frc`，而第一对象使用赋值。若初始化问题修复，第二对象误差与力仍会跨 step 累积，导致位置奖励趋近 0、力项快速饱和。

此外，论文 Table X 的 contact 权重是 1.0，但代码默认 `weight=1.0` 之外又有 `gain=5.0`，实际 active contact 幅值为 5。除非上层 reward group 另有归一化，否则这是额外的论文-配置差异。

## 5. AdaptManip

论文：[PDF](https://arxiv.org/pdf/2602.14363)；项目页：[AdaptManip](https://morganbyrd03.github.io/adaptmanip/)。截至审计日期，项目页和论文未给出官方代码链接，无法做代码级复现核对。

### 5.1 manipulation reward

论文 Eq. (2) 可整理为：

```text
r = r_loco
  + omega_kin [ exp(-|psi_robot-psi_box|)
                + exp(-4 ||p_hand_err||)
                + exp(-1.5 ||p_root_err||) ]
  + omega_box [ exp(-2||p_box-p_des||_1 - ||q_box-q_des||_1)
                + exp(-||v_root-v_box||_2) ]
  + omega_con clamp(sum_h ||f_con,h|| I_box, 0, 1)
  - omega_con sum_h min(0, v_hand,z - v_box,z)
```

变量：

- `r_loco`：底层 locomotion reward。
- `omega_kin`：kinematic tracking 组权重。
- `psi_robot, psi_box`：robot 与 box 的 yaw。
- `p_hand_err`：手相对期望抓持位置的误差。
- `p_root_err`：robot root 相对 box/期望 root 的位置误差。
- `omega_box`：box stabilization 组权重。
- `p_box, p_des`：当前/期望 box 位置。
- `q_box, q_des`：当前/期望 box orientation 表示。
- `v_root, v_box`：root 与 box 线速度。
- `h`：手的索引。
- `f_con,h`：第 `h` 只手与 box 的接触力。
- `I_box`：论文未进一步明确定义，按上下文应为 box contact 有效性/方向 mask。
- `v_hand,z, v_box,z`：手与箱体的竖直速度分量。
- `omega_con`：contact/slip 组权重。

### 5.2 需要谨慎解读的地方

- 文字声称鼓励 symmetric bimanual contact force，但公式只有 `sum_h ||f_con,h||`，并没有 `|F_L-F_R|` 或方差项；一只手承担全部力也可得到相同总和。文字与公式并不完全对应。
- 文字声称抑制 tangential slip，但公式只使用 `z` 方向速度差，不是完整接触切平面的相对速度。
- `-sum min(0, v_hand,z-v_box,z)` 的符号会在手比箱体向上更慢时产生正值；它更像单向支撑/掉落约束，不是对称的 slip penalty。
- `clamp(sum force,0,1)` 若输入以 N 为单位会在约 1 N 时饱和；论文没有交代是否归一化，也没有给出 `omega_*` 数值。由于没有代码，无法消除该歧义。

### 5.3 critic privileged observation

这是四篇里和当前仓库最接近的设计。论文写为：

```text
o_critic,t = [o_actor,t-2:t, o_priv,t-2:t]
```

其中 `o_priv` 包括 ground-truth box 6D pose `X_box`、线速度 `v_box`、角速度 `omega_box`、hand contact force `f_hand`、box contact force `f_box`。它采用三帧 critic history；当前仓库则是 current-frame 143-D critic。

## 6. Sim-to-Real Learning for Humanoid Box Loco-Manipulation

论文：[PDF](https://arxiv.org/pdf/2310.03191)。未发现作者提供的官方代码仓库。

### 6.1 总体奖励形式

```text
R = sum_i w_i exp(-r_i)
```

这里多数 `r_i` 是 cost。pickup 分成：

- contact phase：双手移动到箱体两侧，`t_contact=100` policy steps，即 2 s。
- lift phase：把箱体移动到目标位置，`t_lift=3.5 s`。

三段手目标点：1.5 s 时离箱体侧面 10 cm；2 s 时到达侧面；3.5 s 时随箱体到目标位置。

### 6.2 接触与力项

论文给出：

```text
r_contact = ln(0.05 c_left_hand,box + 0.05 c_right_hand,box)
w_contact = I[t >= t_contact]

r_table      = F_table,box
r_hand_force = ||F_left_hand,box|| + ||F_right_hand,box||
r_box_acc    = ||a_box||
```

变量：

- `c_hand,box`：hand-box contact indicator，论文定义为接触时 1。
- `F_table,box`：桌面对 box 的接触力。
- `F_hand,box`：手与 box 的接触力向量。
- `a_box`：box acceleration。
- `I[t>=t_contact]`：阶段门控，防止提前碰箱。

权重：Table force 0.05、Hand force 0.05、Box acceleration 0.05。作者还在 contact countdown 后 0.5 s 仍未双手接触时终止，在 pickup countdown 后 0.5 s box 仍接触桌面时终止。

这里存在论文内部数学不一致：若 `c=1` 表示接触，且总奖励使用 `exp(-r_contact)`，那么双手接触得到 `exp(-ln 0.1)=10`，单手接触得到 20，无接触则趋于无穷，方向完全反了。很可能公式漏了负号或 indicator 定义/总奖励实现与文中不同。没有代码，不能擅自替作者修正。

在 walking-with-box 中，作者继续使用 `r_box_force=||F_L||+||F_R||` 作为 force cost，同时用“任一手失去接触即终止”保证最小抓持。这形成一个清晰的工程意图：**在不掉箱的前提下尽量用小力，并抑制 box acceleration/冲击**。

## 7. 当前 PhysHSI CarryBox 仓库审计

关键文件：

- `legged_gym/legged_gym/envs/g1/carrybox_config.py`
- `legged_gym/legged_gym/envs/g1/carrybox.py`
- `legged_gym/legged_gym/scripts/validate_carrybox_phase_a.py`

### 7.1 observation 现状

- actor：`738 = 6 x 123`，6 帧历史。
- critic：143 维 current-frame privileged observation。
- critic base：126 维。
- interaction privileged tail：17 维：
  - `box_lin_vel_local`：3。
  - `box_ang_vel_local`：3。
  - `left_hand_net_contact_force_local`：3。
  - `right_hand_net_contact_force_local`：3。
  - `box_net_contact_force_local`：3。
  - `left_hand_contact_flag`：1。
  - `right_hand_contact_flag`：1。

force flag 阈值是 1 N；各向量乘 scale 后统一 clamp 到 `[-10,10]`。

### 7.2 reward 现状

`carryup_task` 当前为：

```text
hand2object_position_reward = exp(-3 * ||mean(p_left,p_right)-p_box||^2)
box_carryup_reward          = exp(-3 * relu(target_box_height-z_box))
```

问题：

- 双手只看均值，无法区分“左右手分别在箱体两侧”和“两只手挤在同一侧”。
- 没有 contact mask、接触力下限、过力、双手平衡或滑移。
- `hand_contact` 配置为 0，因此已有 contact tensor 没有进入 task return。
- critic 看到了 force 不等于 actor 会自然学会接触；它只改善 value/advantage 估计。必须在 reward 中把接触质量和任务收益建立因果关系。

### 7.3 force 信号质量

当前 `left/right_hand_net_contact_force` 来自每个 hand rigid body 的 net contact force；`box_net_contact_force` 是整个单刚体 box 的 net force。主要风险：

1. 不是 pair-specific：不能确认力来自 hand-box。
2. box 净力包含桌面、地面、双手等全部接触。
3. 左右夹持力方向相反时，box 净力可能很小，但抓持实际很强。
4. 10 N 截断会使重箱或冲击阶段大量饱和，critic 无法区分 12 N 和 80 N。

因此 reward 最好使用 filtered hand-box contact。如果 Isaac Gym 当前接口拿不到 pair force，第一版可以用“手靠近对应 box surface 且 wrist net force 超阈值”的 gated proxy，但必须在日志和文档中明确它不是接触力真值。

## 8. 推荐设计

### 8.1 明确左右接触目标

在 box frame 定义：

```text
p_L^obj = [x_c, +s_y/2 - delta_y, z_c]
p_R^obj = [x_c, -s_y/2 + delta_y, z_c]
p_i^w   = p_box^w + R_box^w p_i^obj
d_i     = ||p_hand,i^w - p_i^w||
```

这样替代当前“双手均值到箱心”的奖励。`s_y` 来自 `_box_size[:,1]`，天然适配尺寸随机化。

### 8.2 接触主奖励

对 `i in {L,R}`：

```text
r_pos,i = exp(-relu(d_i-epsilon_pos)/sigma_pos)
r_minF,i = exp(-relu(F_min-f_n,i)/sigma_force)
r_grasp = mean_i I_i^des * r_pos,i * r_minF,i
```

- `I_i^des`：当前阶段是否需要该手接触。
- `f_n,i`：沿 box surface inward normal 的法向接触力，优先使用 hand-box filtered force。
- 如果只有 wrist net force，先以 `||F_hand,i||` 做 proxy，并用 `d_i<epsilon_gate` 门控。

### 8.3 必须补的安全/稳定项

```text
p_over = mean_i I_i^des * relu(f_n,i-F_max)^2 / F_max^2
r_sym  = exp(-|f_n,L-f_n,R|/sigma_sym)

v_rel,i = v_hand,i - [v_box + omega_box x (p_i-p_box)]
v_tan,i = (I - n_i n_i^T) v_rel,i
p_slip  = mean_i I_i^des * ||v_tan,i||^2

p_impact = mean_i relu(||F_i,t||-||F_i,t-1||-DeltaF_max)^2
```

这几项分别解决：过力、单手独占、切向滑移、暴力撞击。它们弥补了 HDMI/HAIC 只设下限、不惩罚过力的缺口。

### 8.4 阶段门控

- approach：只开 `r_pos`，禁止早碰撞或给 early-contact penalty。
- grasp/lift：开 `r_grasp + r_sym - p_over - p_slip - p_impact`。
- carry：保持 force band 与 slip/symmetry，并继续 box pose/height/goal tracking。
- place/release：确认 box 已被平台支撑后，将 `F_min` 平滑降到 0，并奖励双手释放；否则策略会一直夹箱。

不要全程恒定奖励 contact，否则会与 approach、put-down、release 阶段直接冲突。

### 8.5 第一组可用超参数

以下只作为 normalized reward 的起点，随后根据日志校准：

```text
epsilon_pos = 0.04~0.06 m
sigma_pos   = 0.08~0.15 m
F_min       = 3~5 N / hand
F_max       = 25~35 N / hand
sigma_force = 4~8 N
sigma_sym   = 5~10 N
contact_on  = 1.5 N
contact_off = 0.8 N    # hysteresis
```

先把每个 reward term 规范到约 `[0,1]`，再设顶层权重。当前代码会把 scale 乘 `dt`，配置中的权重应按“每秒贡献”理解，不能直接照搬 Isaac Lab/论文的数值。

### 8.6 critic privileged observation 怎么用

第一版无需扩 critic 维度：现有 box pose/task obs、box velocity、hand force 和 contact flag 已足够让 critic 估计 contact-aware value。actor 保持 738 维、仍只依赖可部署信息。

优先修改信号质量，而不是盲目加维度：

1. 将 hand force 换成 hand-box pair force。
2. 将 force 旋转到 box frame，并拆成 normal/tangent，而不是仅给 xyz 净力。
3. 将硬 clamp 10 改为可标定压缩，例如 `tanh(F/F_scale)`，或把 scale 设为 `1/20~1/50` 后 clamp。
4. 如果仍有 value 抖动，再增加 2~3 帧 interaction history，参考 AdaptManip；不要一开始同时扩大 observation、改 reward、改 curriculum。

可以考虑的新 privileged tail：

```text
[box_v(3), box_omega(3),
 fL_normal(1), fL_tangent(2), fR_normal(1), fR_tangent(2),
 contact_L/R(2), d_L/R(2), slip_L/R(2)]
```

## 9. 建议实施顺序与消融

1. **信号验证**：固定姿态分别让左手、右手、box-table 接触，确认 force channel 与符号；记录 mean/p95/p99、饱和率和错误 contact rate。
2. **Phase A**：只把 hand target 从箱心均值改为左右 surface targets；保持 force reward 关闭。
3. **Phase B**：加入 `r_pos * r_minF` 和 contact hysteresis；先不加 overforce/slip。
4. **Phase C**：加入 `p_over`、`r_sym`、`p_slip`、`p_impact`。
5. **Phase D**：再扩大 mass/friction/size randomization，并处理 release。

最小消融矩阵：

- baseline。
- `+ surface target`。
- `+ minimum force`。
- `+ force band`。
- `+ slip + symmetry`。
- `+ critic force privileged info` 与去掉该 privileged info 的对比。

必须记录的指标：双手/单手 contact rate、接触建立时间、lost-contact rate、左右力差、force p95/p99、过力占比、切向滑移速度、box acceleration peak、box drop rate、任务成功率。只看 episode reward 无法判断策略是否真的学会稳定抓持。

## 10. 最终建议

当前仓库最适合从 **HDMI/HAIC 主项 + AdaptManip 稳定项 + Digit 安全项** 的组合开始：

```text
contact correctness = target position x minimum normal force
grasp stability     = force symmetry + low tangential slip
hardware safety     = over-force + force-rate/impact penalty
task sequencing     = phase-dependent activation and release
```

不要直接奖励 `box_net_contact_force`；先解决 pair-specific force 或建立带距离门控的 wrist-force proxy。否则 reward 很容易把桌面支撑力、地面碰撞或双手相消后的净力误认为抓持质量。
