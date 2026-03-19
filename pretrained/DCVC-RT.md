```markdown
# DCVC-RT 重點整理

## TL;DR

DCVC-RT 是一個以 **real-time / low-latency neural video codec** 為目標設計的模型。  
它的核心不是單純追求低 FLOPs，而是更重視 **operational complexity**（例如 memory I/O、latent 大小、module 數量、操作頻率），因此整體架構和許多傳統 neural video codec 很不一樣。

它的幾個核心判斷如下：

- **是 causal / online codec**
  - 只依賴過去已解碼的 reference，不看未來 frame
  - 原理上適合 streaming / realtime video transmission
- **傳輸的不是 raw token / raw embedding**
  - 真正送出去的是 **entropy-coded bitstream**
- **沒有原生設計成 token importance split**
  - 沒有明確的「重要 token / 不重要 token」標記讓你直接丟棄
  - 但有 **frame hierarchy**，以及可被視為比較值得保護的 **z / header / I-frame / anchor frame**
- **官方 pretrained model 有兩個**
  - 一個是 **I-frame image model**
  - 一個是 **P-frame video model**

---

# 1. DCVC-RT 的關鍵設計邏輯

## 1.1 核心目標：降低 operational complexity，而不只是降低 MACs

DCVC-RT 的主要觀點是：

> 神經影像/視訊壓縮的推論速度瓶頸，不只是 MACs/FLOPs，還包括 memory access、latent 尺寸、module 數量、函式呼叫次數等操作成本。

因此它做了幾個很關鍵的架構取捨。

---

## 1.2 拿掉顯式 motion estimation / motion compensation

很多 learned video codec 都會有明確的 motion branch，例如：

- optical flow / motion estimation
- motion compensation
- motion latent / motion entropy model

但 DCVC-RT **把這些顯式 motion 模組砍掉**，改成：

- 從前一張已解碼 frame 的 feature 中抽 temporal context
- 直接和當前 frame 的編碼流程結合
- 用 implicit temporal modeling 取代顯式 motion coding

### 好處
- 大幅減少 module 數量
- 減少 memory I/O
- 減少操作頻率
- 更容易做到 real-time

### 代價
- 在 **大位移 / 大 motion** 場景下，compression efficiency 可能不如帶顯式 motion 的方法
- 但換來明顯的實際速度優勢

---

## 1.3 使用 single low-resolution latent，而不是 progressive downsampling

DCVC-RT 不走傳統多層 progressively downsampled latent 架構，  
而是偏向：

- 先做 patch embedding
- 在 **單一低解析度 latent 空間** 處理
- 論文描述中屬於 **single-scale latent learning（約 1/8 scale）**

### 這樣做的目的
- 降低 latent-wise memory I/O
- 降低 feature map 操作成本
- 提高實際推論速度

### 直觀理解
不是一直把 feature 在很多解析度之間搬來搬去，  
而是盡量把主要處理都集中在一個較低解析度的 latent 空間中完成。

---

## 1.4 Variable-rate 不是只改 quant scale，而是做成 module bank

DCVC-RT 的 variable-rate 設計不是只有簡單改一個 QP 或 scale。  
它額外引入了不同的 module / vector bank 來支援更細緻的 rate control。

可理解為：

- 同一個模型支援多種壓縮率
- 不只是調 quantization 強度
- 還會調整不同子模組的對應行為

這使它能支援較寬的 bitrate range。

---

## 1.5 Integerization

DCVC-RT 還強調了 **16-bit integerization**。

### 目的
- 提高跨平台解碼一致性
- 降低不同 GPU / 裝置 / 浮點實作差異造成的 bitstream 不一致風險

這對真正要做 deployment 的 codec 很重要。

---

# 2. Model Spec

## 2.1 論文層級整體規格

以下是整理出的主要規格：

| 項目 | DCVC-RT |
|---|---:|
| MACs | **385G MACs** |
| Params | **20.7M** |
| FLOPs | 約 **770 GFLOPs**（若以 1 MAC ≈ 2 FLOPs 粗估） |
| 1080p 編碼/解碼速度（A100） | **125.2 / 112.8 fps** |
| 1080p 編碼/解碼速度（RTX 2080Ti） | **39.5 / 34.1 fps** |
| 4K 編碼/解碼速度（A100） | **35.5 / 29.5 fps** |
| 4K 編碼/解碼速度（RTX 2080Ti） | **11.6 / 9.9 fps** |
| BD-rate vs VTM (fp16) | **-21.0%** |
| BD-rate vs VTM (int16) | **-18.3%** |

### 備註
- **MACs / Params** 是論文可直接對到的規格
- **FLOPs** 多半不是論文直接列出，而是以常見換算方式粗估
- bitrate 不是固定值，而是 **content-dependent + QP-controlled variable-rate**

---

## 2.2 Bitrate 特性

DCVC-RT 並不是固定 bitrate codec，而是：

- 可變碼率
- 受內容複雜度影響
- 受 QP / rate setting 控制

### 特徵
- 單模型支援寬範圍 bitrate
- 在低碼率區表現特別有競爭力
- 可以透過 rate control 測很多 operating points

---

## 2.3 官方實作中的主要 channel 設定

在 video model（P-frame 路徑）中，可整理為：

- `g_ch_src_d = 3 * 8 * 8 = 192`
- `g_ch_d = 256`
- `g_ch_y = 128`
- `g_ch_z = 128`
- `g_ch_recon = 320`

I-frame image model（DMCI）中常見設定則可整理為：

- `N = 256`
- `z_channel = 128`
- `g_ch_enc_dec = 368`

---

## 2.4 主要模組組成（P-frame codec）

P-frame video model 的主要模組可概念化為：

- feature adaptor
- feature extractor
- encoder
- hyper encoder
- hyper decoder
- temporal prior encoder
- y prior fusion
- y spatial prior
- decoder
- reconstruction network

---

# 3. 傳輸的 output 到底是什麼？

## 3.1 不是 raw token / embedding，而是 bitstream

DCVC-RT 內部當然有 latent / feature，例如：

- `y`
- `z`
- temporal context
- reconstructed feature

但真正輸出到傳輸層的不是這些 tensor 本身，  
而是：

> **量化後 latent 經 entropy coding 之後形成的 binary bitstream**

所以比較準確的說法是：

- **內部表示**：latent / feature / quantized symbols
- **外部傳輸**：bitstream

---

## 3.2 P-frame 大致流程

P-frame 可概念化為：

1. 取前一張已解碼 reference 的 temporal context
2. 對當前 frame 做 encoder 得到 `y`
3. 對 `y` 再做 hyper encoder 得到 `z`
4. 先編碼 `z`
5. 再用 prior / context 逐步編碼 `y`
6. 最後封成一個 frame 的 bitstream

---

## 3.3 I-frame 大致流程

I-frame 則是 image codec 路線，概念上：

1. 對單張影像做 encode 得到 latent
2. 建立 hyper latent `z`
3. 先編 `z`
4. 再依 spatial prior 順序逐步編 `y`
5. 最後產生 frame bitstream

---

## 3.4 外層封包結構

外部 bitstream 不只是單純 bytes，還有簡單的 syntax/header，例如：

- nal type
- sps id
- height / width
- qp
- stream length
- payload bitstream

也就是：

> 真正傳輸單位是 **frame payload bitstream + header/syntax**

---

# 4. 中間傳輸 unit 有沒有重要性區分？

## 4.1 直接答案：沒有原生的 semantic token importance 設計

如果你問的是：

> 它有沒有像 LLM token 那樣，明確標註哪些 token 重要、哪些不重要，可以直接保留重要的、丟掉不重要的？

答案是：

**沒有。**

DCVC-RT 並不是這種設計。

它內部雖然有把 `y` 分多個階段去建模和編碼，  
但那主要是為了：

- entropy modeling
- spatial prior / decoding order
- 提升壓縮效率

不是為了讓你隨意丟某些 token 後還能穩定解碼。

---

## 4.2 但它有兩種「實質上的重要性」

### (A) Frame-level hierarchy

雖然沒有 semantic token importance，  
但它有 **frame hierarchy / QP hierarchy**。

也就是某些 frame 會被賦予：

- 更低的 QP
- 更高的品質
- 更高的參考價值

這些 frame 可以視為比較重要，例如：

- I-frame
- anchor-like frame
- low-QP reference frame

---

### (B) `z` 其實很值得保護

DCVC-RT 因為拿掉顯式 motion bits，  
`z` 在整體時空建模中更重要。

所以從工程角度來看，  
若要做 unequal protection，**`z` 是優先值得保護的部分之一**。

---

## 4.3 可不可以直接丟掉「不重要的部分」？

### 以官方實作來說
**不建議直接裁 bitstream。**

原因：

- decoder 預期讀到完整 syntax
- `z` / `y` 的 entropy-coded data 有嚴格順序
- 隨便砍掉後半段通常會導致整張 frame 無法正確解碼

所以它不是天然適合做：

- raw latent drop
- arbitrary token drop
- 部分 bitstream 裁剪後還能 graceful degradation

---

## 4.4 如果你想做無線傳輸 / DRB priority，怎麼切比較合理？

如果是站在 wireless / modem / unequal protection 角度，  
比較合理的切法反而是：

1. **header / SPS**
2. **z stream**
3. **I-frame / anchor frame**
4. **其餘 P-frame payload**

也就是：

> 不要把它想成「語義 token」分層，  
> 而是想成「syntax / reference hierarchy / hyperprior importance」分層。

---

# 5. 它是 causal 嗎？

## 5.1 結論：是，屬於 causal / online codec

DCVC-RT 對當前 frame 的 temporal context，  
來自 **過去已解碼的 reference / feature**，而不是未來 frame。

所以從依賴關係來看：

- **只看 past**
- **不看 future**
- **不是雙向參考 B-frame 型態**

因此它可以視為：

- causal
- online
- 適合 low-delay streaming 的方向

---

## 5.2 為什麼這點重要？

因為如果你想拿去做：

- streaming video
- realtime video transport
- video call / low-delay communication

那最先要確認的就是：

> 它會不會依賴未來 frame？

DCVC-RT 的答案是 **不依賴**，  
所以方向上是可行的。

---

# 6. 它適合 realtime video call 嗎？

## 6.1 原理上適合

從 codec 結構來看，它是適合 low-delay / realtime video coding 的：

- causal
- 沒有 future dependency
- real-time throughput 很強
- 設計上就是偏 practical real-time codec

---

## 6.2 但「適合即時傳輸」不等於「直接可拿來做 video call」

這裡要分清楚：

### 它已經具備的
- causal predictive coding
- 很快的 codec throughput
- 適合 low-delay encode/decode

### 它還沒自然解決的
- packet loss robustness
- error propagation handling
- NACK / FEC / retransmission strategy
- jitter buffer interaction
- RTC stack integration

---

## 6.3 為什麼 video call 仍有挑戰？

因為它是 reference chain：

- 每個 P-frame 依賴前面已解碼結果
- 一旦某張 frame 丟失或損壞
- 後續 frame 可能持續受影響
- 直到 refresh / reset / 新 I-frame 才恢復

這是很多 predictive codec 都會遇到的問題。

---

## 6.4 實務上的建議

若你想拿 DCVC-RT 做 realtime video call / wireless streaming，  
比較合理的工程方向是：

- 採 low-delay P 結構
- 定期插入 refresh / I-frame
- 對 header / `z` / anchor frame 加強保護
- 將 codec throughput 與 network robustness 分開設計

也就是：

> DCVC-RT 比較像「可即時、causal 的 neural predictive codec」  
> 而不是「天然抗丟包的通話 codec」。

---

# 7. 為什麼 pretrained model 有兩個？

官方提供兩個 checkpoint：

- `cvpr2025_image.pth.tar`
- `cvpr2025_video.pth.tar`

## 7.1 這不是二選一，而是同一套系統的兩個子模型

### `cvpr2025_image.pth.tar`
- 給 **I-frame / intra-frame**
- 可理解成 image codec
- 對應 **DMCI**

### `cvpr2025_video.pth.tar`
- 給 **P-frame / inter-frame**
- 可理解成 predictive video codec
- 對應 **DMC**

---

## 7.2 實際運作方式

在一段影片中：

- 第一張 frame 或 refresh frame
  - 用 image model 編碼
- 其餘 frame
  - 用 video model 做 predictive coding

因此兩個 checkpoint 是互補的。

---

## 7.3 直觀理解

可以把它想成：

- **image model = 關鍵幀編碼器 / reset 點**
- **video model = 連續幀預測編碼器**

所以 pretrained model 有兩個是非常合理的，  
不是因為版本分裂，而是因為它本來就是兩段式設計。

---

# 8. 從無線傳輸 / modem 角度的解讀

如果你的目標是：

- priority split
- unequal protection
- DRB mapping
- HARQ / FEC 設計

那對 DCVC-RT 最合理的理解是：

## 8.1 它的傳輸單位是 bitstream，不是 semantic token

所以不要把它當成：

- LLM token stream
- 顯式可裁剪的重要度 token 序列

---

## 8.2 較適合保護的對象

比較合理優先保護的部分包括：

1. **SPS / header**
2. **I-frame**
3. **low-QP / anchor frame**
4. **z / hyperprior 相關 bitstream**
5. **其他 P-frame payload**

---

## 8.3 較不適合的想法

比較不適合直接套在 DCVC-RT 上的想法包括：

- 「每個 latent channel 都有語義重要度，可直接分類送不同 DRB」
- 「丟掉一部分 entropy-coded payload 仍能平滑退化」
- 「像 token pruning 一樣直接刪一些 unit 也能正常解碼」

這些在官方設計上都不是天然成立的。

---

# 9. 最後結論

## 9.1 一句話總結 DCVC-RT

DCVC-RT 是一個：

> **以 operational complexity 為核心優化目標的 causal real-time neural video codec**

它透過：

- 移除顯式 motion 模組
- 使用單一低解析 latent
- 使用 variable-rate module bank
- 支援 integerization

來達成很強的 real-time throughput。

---

## 9.2 對你問題的直接回答

### (1) 它的傳輸 output 是什麼？
- **最終傳的是 entropy-coded bitstream**
- 不是 raw token / raw embedding

### (2) 中間 unit 有沒有重要性區分？
- **沒有原生 semantic token importance**
- 但有：
  - frame-level importance
  - `z` / header / I-frame 等較值得保護的部分

### (3) 可不可以保留重要的、丟掉不重要的？
- **官方格式下不能直接這樣做**
- 若要做，需要你自己重設 packetization / syntax split

### (4) 它 causal 嗎？
- **是**

### (5) 適合 streaming / realtime video call 嗎？
- **原理上適合**
- 但還要額外補：
  - packet loss robustness
  - refresh
  - FEC / NACK / unequal protection

### (6) 為什麼有兩個 pretrained model？
- 因為它本來就由：
  - **I-frame image model**
  - **P-frame video model**
  兩個子系統組成

---

# 10. 適合直接貼到筆記中的超短版

## DCVC-RT 一頁摘要

- **定位**：real-time / low-delay neural video codec
- **核心設計**：
  - 重視 operational complexity，不只看 FLOPs
  - 拿掉顯式 motion estimation / compensation
  - 用過去已解碼 feature 做 implicit temporal modeling
  - 採 single low-resolution latent
- **Spec**：
  - 385G MACs
  - 20.7M params
  - 約 770 GFLOPs（粗估）
- **輸出**：
  - 最終傳的是 **entropy-coded bitstream**
  - 不是 raw token / embedding
- **重要性**：
  - 沒有原生 token importance split
  - 但 I-frame / anchor frame / z / header 比較值得保護
- **Causal**：
  - 是，只看 past，不看 future
- **適合即時傳輸嗎**：
  - 是，方向上適合
  - 但還要自己補 packet loss robustness / refresh / FEC / NACK
- **兩個 pretrained model**：
  - `image.pth.tar`：I-frame
  - `video.pth.tar`：P-frame

```

如果你要，我可以再幫你整理成 **更精簡的 README 風格版本**，或是改成 **「DCVC-RT for wireless/video call」專用筆記版 markdown**。
