"""时空笔画架构配置（src/st/config_st.py）。

从零重构 —— 设计契约见 docs/22（已批准 2026-08-20），旧库 smartink_torch/ 仅作只读参考。
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class STConfig:
    """时空模型超参。默认值沿袭当前规范 config（pretrained_model/config_quickdraw.json 量级）。"""

    # ── 坐标 ──
    coord_dim: int = 3            # (x, y, Δt)；pen 通道已删除（docs/22 §2-1）。
                                  # A2 复现模式（--repro_a2）降为 2：(x,y)，无 Δt
    dt_jitter: float = 1e-2       # 切片任务 Δt 抖动幅度（D-e；docs/21 §5.1 第三对策）
    temporal_resampling_factor: int = 1   # 长笔 stride 抽稀（>100 点按 factor 抽，
                                  # 对齐旧 pp_temporal_resampling）；A2 复现模式=2

    # ── 笔画编码器 E ──
    enc_d_model: int = 96
    enc_layers: int = 6
    enc_heads: int = 4
    enc_dff: int = 384
    enc_dropout: float = 0.0
    latent_units: int = 16
    max_stroke_len: int = 200     # 笔画内点序列最大长度（正弦 PE 表长）

    # ── 曲线解码器 D ──
    dec_layers: int = 4
    dec_hidden: int = 512
    # 2026-10-06：新AE使用连续τ正弦编码+FiLM；旧checkpoint缺省仍为concat。
    dec_tau_mode: str = "concat"  # concat=旧[τ,λ]；film=λ主干、τ编码调制每层
    dec_tau_pe_dim: int = 64      # sin/cos成对；固定频率，无查表或整数τ
    dec_tau_max_cycles: float = 32.0  # τ∈[0,1]的最高周期数；最低0.5周期区分两端
    gmm_dec: int = 20             # 笔画点 GMM 分量数
    n_t_samples: int = 4          # 每笔每步随机 τ 采样数（端点 τ∈{0,1} 另加）
    endpoint_loss_w: float = 0.1  # D-d：端点辅助损失权重。2026-08-22 用户裁定撤回：
                                  # 端点单独监督可能扭曲 λ 空间自然成形（AE 先学定位
                                  # 端点而非建模整曲线）；A2 复现模式=0

    # ── 关系模型 R（双模型，D-a）──
    r_d_model: int = 160
    r_layers: int = 8
    r_heads: int = 8
    r_dff: int = 640
    r_dropout: float = 0.0
    gmm_r: int = 10               # R 输出 GMM 分量数
    r_pe_mode: str = "concat"     # R 位置编码（D27，2026-08-22 用户理论）：
                                  #   rope       = 片级 RoPE 旋转嵌入（D-b，已证伪：旋转固定
                                  #                λ 点云致混叠，两片任务的 pred 回弹根因）
                                  #   none       = 无位置编码（消融臂：预期恢复）
                                  #   rope_stroke= 逐笔 RoPE（对照臂：预期复现 CoSE 论文
                                  #                "顺序编码版性能略降"现象）
                                  #   concat     = 片 PE 拼接到输入特征（修复架构：额外自由度
                                  #                承载时序，不动嵌入流形）
                                  #   page       = 2026-10-02 新页协议：只拼当前页 PE，
                                  #                无目标页查询、无结束二分类头
    slice_pe_dim: int = 32        # concat/page 的片 PE 维度；page 按页索引直接计算
    max_context_strokes: int = 1024  # 2026-10-02：新管线的笔画上下文预算（含控制符）；
                                     # 旧 A1' 路径暂不按此字段截断或强制长度
    end_head_w: float = 1.0       # 片末二值头 BCE 权重（终止符 e 的退化形态：与 R_pos
                                  # GMM 头并联的轻量 logit 头，"下一笔是片末"→1）
    rope_theta: float = 10000.0   # 片级 RoPE 基数（D-b）
    max_slices: int = 64          # 片索引上限（RoPE 无需外推限制，仅兜底断言）

    # ── D28：单 trunk 双头（2026-08-22 用户裁定，放弃双塔）──
    # 根因：并行架构下把目标起点绑在特定位置的 token 上，其它位置 attend 时拿到
    # 被无关查询污染的内容（原版无掩码+单预测，可以把查询平铺到所有 token）。
    # 方案1：同一 R 一次前向并联输出位置 GMM + 嵌入 GMM，无查询注入；
    # 方案2（备选）：位置头挂第4层、嵌入头挂第6层，强迫位置信息先凝聚。
    # 都炸则放弃并行化回退旧管线（修 pen 通道）。
    r_dual_head: bool = False     # 单 trunk 双头开关
    r_mid_head_layer: int = 0     # 方案2：位置头挂的中间层（0=不启用）
    r_head_mlp: bool = False      # 输出头前加缓冲 MLP(512,256)+ReLU+LN（2026-08-23
                                  # 用户裁定：还原 v4 被我省略的输出通路；实测梯度
                                  # 冲突不成立 cos≈+0.08，病灶收敛到输出级容量）

    # ── 阶段2（docs/24 §3）：旧双塔 + 高层条件注入（2026-08-23 已批准）──
    # R_pred 的目标起点查询不送输入层，改为在第 K 层 Block 前加性注入
    # （高层隐向量已聚合全局语义，此处注意力系统性偏向自身与既有解释不冲突）。
    # 修正案：阶段2只训嵌入塔 R_pred，r_skip_pos=True 时 R_pos 前向整个跳过。
    late_query_layer: int = 0     # 查询注入层（0=旧行为：注入输入层 token）
    r_skip_pos: bool = False      # 跳过 R_pos 前向与位置损失（阶段2单塔诊断）

    # ── D29（2026-08-23 用户裁定）：GMM 头条件化 ──
    # 原架构里目标位置出现两次：输入侧平铺（空间注意力偏置）+ 拼进 MLP→GMM 头
    # （真正的条件解码器）。并行+因果+shuffle 下输入侧注入只剩病态偏置，故
    # transformer 侧零注入，目标起点直接拼到 GMM 头输入，MLP 头加深
    # r_head_extra 层（预期稍差于 v4：隐向量缺空间注意力偏置）。
    r_head_query: bool = False    # 目标起点拼到查询塔 GMM 头输入（替代一切注入）
    r_head_extra: int = 0         # 缓冲 MLP 追加的 512 宽隐层数（加深 GMM 头）

    # ── 计划1 统一嵌入（2026-08-23 用户"最后的冲锋"，已批准）──
    # 放弃形状/位置解耦：AE 输入输出保留绝对画布坐标，λ 统一编码位置+形状；
    # DCT 辅助头（目标平移不变）监督纯形状，防位置语义过度混淆。
    # 用户论证：等价于给轨迹加"永不监督的自由行程"前缀，函数类不变、不难本质化。
    unified_emb: bool = False   # 统一嵌入开关（仅支持 coord_dim=2）
    dct_coeffs: int = 0         # >0 启用 DCT 纯形状辅助头（每通道 k=1..K，共 2K 维）
    dct_weight: float = 0.1     # DCT MSE 权重（对齐 docs/07 旧实验 α=0.1）
    dct_hidden: int = 64        # DCT 头隐层宽（用户裁定：取 v1(3×128) 与 v2(1×32)
    dct_layers: int = 2         #  之间，默认 2×64——v2 单层惩罚可能过强）
    dct_dc: bool = False        # B 臂（2026-09-24 用户裁定）：目标末尾追加质心
                                # (x̄,ȳ)（直流分量等价量；统一画布下=笔画位置），
                                # 显式位置+形状解耦监督，目标维度 2K→2K+2

    # ── GMM 数值契约（沿袭 output.py:431-434，加方差下限）──
    sigma_clamp_lo: float = -15.0
    sigma_clamp_hi: float = 5.0
    sigma_eps: float = 1e-4       # 标准差下限（Δt 零方差退化对策其一；docs/21 §5.1）
    sigma_eps_r: float = 1e-2     # R 头专用更严下限（A1 崩塌事故对策：给 pred NLL 封顶，
                                  # 防 σ→eps 锐化梯度吃光分组裁剪预算；docs/22 §13）
    r_sigma_floor_mode: str = "add"  # 2026-10-09：旧cfg缺省add；新T1显式quadrature，
                                    # σ=√(exp(clamp(raw))²+sigma_eps_r²)，不改AE头
    gmm_head_fp32: bool = False   # 2026-10-02：新 FP16 训练显式用 FP32 算 GMM 线性头；
                                  # False 保留旧 checkpoint 的 autocast 行为

    # ── 在线增强（对齐旧管线 affine_prob=0.3；A1 崩塌事故后补回，docs/22 §13）──
    affine_prob: float = 0.3

    # ── 采样与监督（docs/22 §5）──
    w_first: float = 3.0          # 每片首笔 loss 加权（D-c 修订：过采样→加权）
    p_shuffle_graphic: float = 1.0   # riffle 触发率（2026-08-21 用户裁定 0.5→1.0：
                                      # 触发后 ½ 分两段 × ½ 段序不变 = ¼ 完全有序，
                                      # 部分乱序占主导，对标旧 A1 的 16 有序+16 随机）
    first_stroke_prior: bool = False  # 首笔先验（2026-09-25 用户裁定，治开局模式
                                      # 坍塌：纯文字上下文下 GMM 近均匀摊 29/32 分量，
                                      # greedy 恒锁"最平庸通用笔"）：构造样本时 50%
                                      # 把弧长最长笔放图形序列首位、50% 真实首笔，
                                      # 首位豁免 riffle 打乱

    # ── 预测专用输入编码器（2026-09-27 用户构想，"目标机/输入机"分离）──
    # 随机初始化的同架构编码器接 R 输入、不截梯度端到端训练；预训练冻结编码器
    # （self.encoder）只产预测目标 λ。输出接口补 LayerNorm + 直投影到 r_d_model
    # （不过 16 维瓶颈）；loss_weight/targetable/loss 形状契约不变。
    free_input_encoder: bool = False
    p_shuffle_text: float = 0.0      # 文字片：对照消融项 {0, riffle}，默认关
    shuffle_max_segments: int = 3    # riffle 的 n+1 段上限参考
    class_balanced: bool = True      # BalancedClassBatchSampler（无限流均衡，无队尾余量）

    # ── 终止符 e（D25；docs/22 §11/§14）──
    use_terminator: bool = False     # 2026-08-21 用户裁定撤下：V100 job 512453 上
                                     # pred_g 于 ~2-4k 步回弹且 pred_e 发散振荡——
                                     # e 的静止常数目标诱导 R_pred 过度收紧 σ，
                                     # 在持续漂移的 λ 分布上反复 miss。回退到
                                     # 无终止符的 A1 等价形态。代码保留备查。
    term_pos: tuple = (4.0, 4.0)     # e 起点（raw start 空间，画布外）。2026-08-21 由 [1.5,1.5]
                                     # 改 [4,4]：affine 增强恢复后 [1.5,1.5] 有 0.15% 起点冲突，
                                     # [4,4] 在 r<0.3 内 0/23821（增强后 y≲2.5 够不到）
    term_radius: float = 0.08        # e 环半径（raw 单位；画布幅度 ~1–2.6）
    term_points: int = 8             # e 笔画点数

    # ── 训练 ──
    stop_predictive_grad: bool = True   # 不变量 #2：R 梯度不回传 E
    freeze_ed: bool = False             # 冻结 E/D（只训 R，decoder 前向跳过）：
                                        # 2026-08-22 用户的"移动靶"诊断探针——
                                        # λ 冻结后 pred 应开始下行，否则问题在 R 自身
    pos_loss_w: float = 1.0             # 位置损失权重（2026-08-23 用户：位置信号
                                        # 淹没嵌入信号的嫌疑 → 试 0.005）
    chord_condition: bool = False       # chord 向量进 decoder（消融开关，docs/21 §5.2c）

    # ── 混合精度（docs/22 §11：fp16 事故修复方案）──
    amp: bool = True
    amp_dtype: str = "bf16"             # "bf16"（A100+）/ "fp16"（V100，配 GradScaler）/ "off"
    # GMM 头部与 NLL 恒在 fp32（model.py 内置，无需配置）

    # 2026-10-14批准：R文字形状/逐笔空格辅助监督；旧cfg默认关闭。
    r_text_aux_enabled: bool = False
    r_text_aux_classes: int = 116
    r_text_aux_hidden: int = 256
    r_text_aux_weight: float = 0.0  # 首次在计算节点校准，完整恢复继承实际值。
    # 2026-10-27：训练用对象类别及可见笔画目标匹配；旧cfg默认关闭。
    r_grounding_enabled: bool = False
    r_grounding_classes: int = 345
    r_grounding_key_dim: int = 128
