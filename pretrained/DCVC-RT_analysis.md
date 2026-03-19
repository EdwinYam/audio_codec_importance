# DCVC-RT 深度分析筆記

> 整理自對 DCVC-RT 論文、官方 repo 與實作程式碼的交叉比對分析。

---

## 目錄

1. [關鍵設計邏輯](#1-關鍵設計邏輯)
2. [Model Spec](#2-model-spec)
3. [傳輸的 Output 是什麼？](#3-傳輸的-output-是什麼)
4. [中間傳輸 Unit 有沒有重要性區分？](#4-中間傳輸-unit-有沒有重要性區分)
5. [Causal 性質與 Realtime Streaming 適用性](#5-causal-性質與-realtime-streaming-適用性)
6. [為何有兩個 Pretrained Model？](#6-為何有兩個-pretrained-model)
7. [Wireless Transport 觀點總結](#7-wireless-transport-觀點總結)

---

## 1. 關鍵設計邏輯

### 核心觀念：降「operational complexity」而非只降 FLOPs

論文主張 NVC 的速度瓶頸很多時候不是 MACs 本身，而是 **memory I/O、tensor size、module 數量、函式呼叫頻率**等操作成本。Fig. 3 顯示：
- 單純降計算量，推論時間不會等比例下降
- 高計算複雜度下，**latent size** 更關鍵
- 低計算複雜度下，**module 數量** 更關鍵

### 三個關鍵架構轉向

#### (1) 拿掉顯式 Motion Estimation / Motion Compensation → Implicit Temporal Modeling

- 不用傳統 learned video codec 常見的 motion branch
- 從前一個已解碼 reference 的 feature 中抽 temporal context，直接和目前 frame 的編碼路徑一起處理
- 少掉大量 motion-related modules，降低 operation frequency
- 雖然大 motion 的 BD-rate 稍差，但編碼速度快了 **3.4 倍**
- Scene change 反而更有優勢

#### (2) Single Low-Resolution Latent（非 Progressive Downsampling）

- 先做 patch embedding，把 frame 放到單一低解析度空間處理
- 採用 **1/8 single-scale latent learning**
- 大幅減少 latent-wise memory I/O
- 在公平 MACs 下，1/8 scale 比 progressive downsampling 約快 **3.6 倍**

#### (3) Module Bank Rate Control

- 不是只對 y 做簡單 scale
- 針對不同 QP 準備 **hyperprior module bank**
- 另有 `q_e` / `q_d` / `q_f` / `q_r` 這些 vector bank，分別調整 encoder、decoder、feature extractor、reconstruction network 的振幅
- 單模型可支援更細緻的 variable-rate

### 其他實務設計

- **Model Integerization**：16-bit integerization，確保跨裝置（不同 GPU/平台）編解碼一致，不因浮點數不確定性導致解出不同結果

---

## 2. Model Spec

### 整體 Complexity（論文 Table 3）

| 指標 | 值 |
|---|---|
| **MACs** | 385G |
| **Params** | 20.7M |
| **BD-rate vs VTM** | fp16: -21.0%, int16: -18.3% |
| **1080p fps (A100)** | enc 125.2 / dec 112.8 |
| **1080p fps (RTX 2080Ti)** | enc 39.5 / dec 34.1 |
| **4K fps (A100)** | enc 35.5 / dec 29.5 |
| **4K fps (RTX 2080Ti)** | enc 11.6 / dec 9.9 |

> 論文沒有直接列 FLOPs，只有 MACs。若用常見換算 1 MAC ≈ 2 FLOPs → 385G MACs ≈ 770 GFLOPs（換算值，非論文原文）。

### Code-Level Channel / Module Spec

**Video Model (DMC) 主要 channel 設定：**

```python
g_ch_src_d = 3 * 8 * 8  # = 192
g_ch_d     = 256
g_ch_y     = 128
g_ch_z     = 128
g_ch_recon = 320
qp_shift   = [0, 8, 4]
```

**Video Model 主要模組：**

| 模組 | 用途 |
|---|---|
| `feature_adaptor_i` | I-frame feature 適配 |
| `feature_adaptor_p` | P-frame feature 適配 |
| `feature_extractor` | 特徵提取 |
| `encoder` | 編碼器 |
| `hyper_encoder` | 超先驗編碼器 |
| `hyper_decoder` | 超先驗解碼器 |
| `temporal_prior_encoder` | 時域先驗編碼器 |
| `y_prior_fusion` | y 先驗融合 |
| `y_spatial_prior` | y 空間先驗 |
| `decoder` | 解碼器 |
| `recon_generation_net` | 重建生成網路 |

**Image Model (DMCI) 預設：**

```python
N           = 256
z_channel   = 128
g_ch_enc_dec = 368
```

### Bitrate Spec

- 不是固定 bitrate codec，而是 **content-dependent + qp-controlled variable-rate**
- 訓練時 QP 在 0~63 間隨機取值
- 單模型支援 wide bitrate range，測試時 `rate_num` 可設 2~64
- 在 UVG 的低碼率區 (< 0.02 bpp) 表現特別強

---

## 3. 傳輸的 Output 是什麼？

### 答案：Bitstream，不是 Token / Raw Embedding

把量化後的 latent 用 entropy coder 壓成 bytes，再加上 header/SPS 後寫出。Arithmetic coding 跑在 CPU，bitstream writing 需要 C++ 支援。

### P-frame 內部流程

```
前一張 reference frame/feature → 取 temporal context
                ↓
當前 frame → encoder → y
                ↓
y → hyper encoder → z
                ↓
z 量化 → entropy encode (先編 z)
                ↓
y → two-step prior → 切成兩段寫入 (再編 y)
                ↓
flush → bit_stream
```

### I-frame 內部流程

```
當前 frame → encoder → y
                ↓
y → hyper encoder → z
                ↓
z 量化 → entropy encode (先編 z)
                ↓
y → 4-step spatial prior → 切成四段寫入 (y_q_w_0 ~ y_q_w_3)
                ↓
flush → bit_stream
```

### Bitstream Container / Syntax

外層封包為類似簡單 NAL-like 結構（來自 `stream_helper.py`）：

```
SPS:
  nal_type + sps_id + height + width + ec_part + use_ada_i

每個 I/P frame:
  nal_type + sps_id + qp + stream_length + bit_stream
```

> 1080p 以上解析度時，會啟用 **two entropy coders**，記在 `ec_part` 裡（throughput 優化，非語義雙層重要性）。

---

## 4. 中間傳輸 Unit 有沒有重要性區分？

### 直接答案

官方 DCVC-RT **沒有**把傳輸單位設計成「語義上有重要/不重要 token」的形式。

- y 切成 2 段或 4 段是為了 entropy model / spatial prior 的解碼順序，**不是可隨意丟棄的 token**
- Decoder 預期讀到完整 bitstream
- 外層把整個 frame payload 當成一個 length-delimited `bit_stream` 寫入

### 但有兩種「實質上的重要性」

#### (a) Frame-level Hierarchy 有重要性差別

- 訓練時在 8-picture group 裡用不同 QP offset：`[0, 8, 0, 4, 0, 4, 0, 4]`
- 有些 frame 被賦予更高品質、更低 QP
- P-frame 也有 `qp_shift=[0,8,4]` 搭配 index map 做層級品質控制
- 這是 **frame 層級的重要性**，不是 latent-symbol 層級

#### (b) z 比一般人想像中重要

- 因為沒有 motion bits，z 在時空建模裡變得更關鍵
- 平均上 z 的 bit 數超過 y 的 10%
- z 的分佈估計不準會明顯影響整體表現

### 能不能「重要的保留，不重要的丟」？

以官方 bitstream 格式：**不建議直接丟**。

- 封包層是「整個 frame 一個 bitstream」
- Decoder 要按順序把 z 和後續 y 全部讀完
- 隨便裁掉後半段 → 整張 frame 大概率無法正常解碼

### 可行的研究方向

| 方向 | 說明 | 可行性 |
|---|---|---|
| Frame-level unequal protection | 先保護 I-frame / 低 QP anchor frame | ✅ 與原本 hierarchical quality 結構相容 |
| Syntax-level unequal protection | 把 z、header、y 的 early pass 單獨 packetize，給不同 protection level | ✅ z 很關鍵，比「隨便丟 y 的某些 channel」更有根據 |
| LLM-style token importance | 當成有語義排序的 token 來丟 | ❌ 不適用，這裡是 quantized latent symbols / entropy-coded syntax elements |

---

## 5. Causal 性質與 Realtime Streaming 適用性

### 是否 Causal？→ 是

- 論文 Fig. 4 明寫：temporal context 來自 **previously decoded latent** f_{t-1}
- 只用過去已解碼的資訊，不需要看未來 frame
- 預設評測用 single intra-frame setting (`intra_period = -1`)，不靠 future B-frame 雙向參考

### 適合 Realtime Video Call 嗎？

**原理上可以，方向是對的**，但不等於直接可用。

#### 適合的點

- 定位為 low latency / real-time coding
- Rate control 是為了動態網路條件與 real communication scenario
- 1080p 在 A100 可到約 125/113 fps（編/解碼）

#### 需要注意的點

| 議題 | 說明 |
|---|---|
| **Past-reference chain** | 每個 P-frame 依賴前面已解碼結果，一旦某 frame 丟失/解碼失敗，後面幾張會受影響，直到 refresh / reset / 新 I-frame。Repo 有 `--reset_interval 64` 週期性重置 |
| **Entropy coding 在 CPU** | Arithmetic coding runs on the CPU，實際寫 bitstream 時 CPU 狀態會影響延遲 |
| **只有 codec throughput** | 沒有完整 RTC stack（capture, packetization, jitter buffer, NACK/FEC, render）的延遲測量 |

#### 用於 Realtime Video Call 的最合理用法

1. 採 low-delay P 流程，不用未來 frame
2. 定期插入 refresh / intra
3. 對 header / 關鍵 frame / hyperprior bits 做較強保護
4. 把 codec throughput 和 network loss robustness **分開設計**

> DCVC-RT 比較像「可即時、causal 的 neural predictive codec」，不是「天然抗丟包的通話 codec」。

---

## 6. 為何有兩個 Pretrained Model？

### 不是兩個版本擇一，而是同一套系統的兩個必要部件

| 檔案 | 對應模型 | 用途 |
|---|---|---|
| `cvpr2025_image.pth.tar` | `DMCI()` | **I-frame / intra-frame** 影像編碼 |
| `cvpr2025_video.pth.tar` | `DMC()` | **P-frame / inter-frame** 視訊編碼 |

### 載入方式（test_video.py）

```python
model_path_i → DMCI()   # i_frame_net
model_path_p → DMC()    # p_frame_net
```

### 編碼流程中的分工

```
Frame 0 (或 intra period refresh)
  → i_frame_net.compress(...)
  → 重建結果 encoded['x_hat'] 放入 P-frame 的 reference buffer

Frame 1, 2, 3, ...
  → p_frame_net.compress(...)
  → 使用前一張的重建結果做時域預測
```

### 從 Realtime Video Call 角度理解

- `cvpr2025_image.pth.tar` = **關鍵幀編碼器 / reset 點**
- `cvpr2025_video.pth.tar` = **連續幀預測編碼器**

> 正常 video coding 兩個都需要；只有在 `force_intra` / all-intra / image-only 時，才可能只用 image model。

---

## 7. Wireless Transport 觀點總結

如果站在 modem / DRB / unequal protection 角度：

| 項目 | 現況 |
|---|---|
| 傳輸單位 | bitstream packet |
| 是否 raw embedding | 否 |
| 有無 token importance flag | 無 |
| 有無 frame hierarchy | 有 |
| z 是否值得特別保護 | 是 |

### 最自然的 Priority 切法

```
Priority 1 (最高): SPS / header
Priority 2:        z stream (hyperprior)
Priority 3:        Anchor / low-QP frames (I-frame, hierarchy 中的 key frame)
Priority 4 (最低): 其餘 P-frame payload
```

> 這個切法與 DCVC-RT 的官方設計邏輯相容，但需要自行修改 bitstream syntax，原版 repo 沒直接提供。
