# LoRA 微调训练日志 (v1)

> Qwen2.5-3B-Instruct | Google Colab T4 | 2026-06-16

## 数据

| 指标 | 值 |
|------|-----|
| 训练样本 | 4,592 |
| 验证样本 | 1,148 |
| 数据版本 | v2（已过滤 gpt 开头样本，零 invalid role 警告） |

## 超参

| 参数 | 值 |
|------|-----|
| 模型 | Qwen2.5-3B-Instruct |
| 微调方式 | LoRA (rank=8, target=all linear) |
| 精度 | FP16 |
| Batch size | 2 × 8 梯度累积 = 16 |
| 学习率 | 5e-5 (cosine decay) |
| Epochs | 2 |
| 优化步数 | 574 (287/epoch) |
| 可训参数 | 14.97M / 3100M (0.48%) |
| 量化 | 无（3B FP16 ≈ 10GB，T4 16GB 够） |

## Loss 曲线

| Step | Epoch | Loss | Grad Norm | LR |
|------|-------|------|-----------|-----|
| 5 | 0.17 | 5.567 | 1.16 | 4.22e-5 |
| — | 0.35 | 4.031 | 1.48 | 4.92e-5 |
| — | 0.52 | 3.852 | 1.57 | 4.63e-5 |
| — | 0.70 | 3.709 | 1.81 | 4.13e-5 |
| — | 0.87 | 3.635 | 2.42 | 3.49e-5 |
| — | 1.05 | 3.482 | 2.62 | 2.76e-5 |
| — | 1.22 | 3.336 | 5.24 | 2.00e-5 |
| — | 1.39 | 3.276 | 2.98 | 1.29e-5 |
| — | 1.57 | 3.219 | 3.34 | 6.90e-6 |
| — | 1.74 | 3.213 | 4.40 | 2.56e-6 |

**Loss 下降：5.57 → 3.21（↓42%），健康。**

## 踩坑记录

1. **`train_bash.py` 不存在** → 新版 LLaMA-Factory 改用 `llamafactory-cli train`
2. **bitsandbytes 崩溃** → T4 不需要 4bit 量化，去掉 `--quantization_bit 4`
3. **BF16 慢（30s/步）** → T4 不支持 BF16 硬加速，换 FP16（6s/步，快 4.6×）
4. **80% 样本以 gpt 开头被跳过** → `build_lora_data_v2.py` 修复，确保每条样本 system 后第一个角色是 human
5. **Colab 空闲断开风险** → 训练期间需偶尔点击页面

## 训练用时

- 总步数：574
- 每步：~6.4s (FP16)
- 总时间：约 61 分钟
- Checkpoint：step-500 已保存

## 下次改进

- 加 `--warmup_steps` 替代已弃用的 `--warmup_ratio`
- 试用 `--loraplus_lr_ratio` 提升 LoRA 效果
- 考虑加 eval loss 监控过拟合

---

# v3 训练日志（7B AutoDL）

> Qwen2.5-7B-Instruct | AutoDL 4090D | 2026-06-17

## 数据

| 指标 | 值 |
|------|-----|
| 训练样本 | 4,592 |
| 验证样本 | 1,148（未加载 eval，省显存） |
| system prompt | `你正在微信上与朋友聊天。`（12 字） |

## 超参

| 参数 | 值 |
|------|-----|
| 模型 | Qwen2.5-7B-Instruct |
| 微调 | LoRA (rank=8, target=all linear) |
| 精度 | BF16 |
| Batch | 1 × 8 梯度累积 = 8 |
| 学习率 | 5e-5 (cosine decay) |
| Epochs | 2 |
| 步数 | 1,148 (574/epoch) |
| 可训参数 | 20.19M |
| 量化 | 无（bitsandbytes 装不上） |

## 结果

| 指标 | v1 3B T4 | v3 7B 4090D |
|------|:---:|:---:|
| Train Loss | 3.21 | 3.39 |
| 速度 | 6.4 s/step | 1.93 s/step |
| 总时间 | 61 min | 37 min |
| 推理效果 | 短回复，身份模糊 | 短回复，身份模糊 |

## 结论

**7B 没带来质变。** 风格迁移瓶颈在数据质量和 system prompt 策略，不在模型大小。

**当前最优方案**：
- 训练时：泛用短 prompt（`你正在微信上与朋友聊天。`）
- 推理时：人格化长 prompt（`你是阿狸...` + 撒娇风格描述）
- 模型：3B 足够，7B 提升边际

## 新增踩坑

6. **AutoDL 无外网 pip install 失败** → bitsandbytes 装不上，7B 不加量化直接跑 BF16
7. **ModelScope 下载大文件反复失败** → 多次重试，支持断点续传
8. **系统盘爆满** → 模型必须下载到 `/root/autodl-tmp/`（数据盘）
9. **终端输入 Unicode 损坏** → 删除字符产生异常空格，tokenizer 崩溃；改用批量脚本测试
10. **700MB 打包 vs 40MB 权重** → 只需下载 `adapter_model.safetensors` + `adapter_config.json`
11. **7B vs 3B 风格迁移无显著差异** → LoRA 学的是数据分布，数据量不够时大模型无优势
