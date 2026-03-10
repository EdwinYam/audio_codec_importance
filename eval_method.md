可以，而且我建議你把這個 PoC 設計成**「通話系統實驗」**，不是單純的 codec 重建實驗。

因為 IMS / MTSI 在乎的不只是 decoder 出來的音質，還包含 **jitter buffer、packet-loss handling、adaptation、端到端互動性**；3GPP 也明講 MTSI 的目標是可預期的 media quality，而 EVS/IMS 世界裡 20 ms framing 與 jitter buffer management 本來就是標準玩法。([ETSI][1])

## 一句話先定義你的研究問題

你真正要回答的問題應該是：

**在固定額外保護成本下，importance-aware 保護是否比 random 保護或簡單 heuristic 保護，更能改善 IMS audio call 在 loss + jitter 條件下的可懂度與主觀品質，而且不把延遲搞炸。**

這句很重要，因為它把「保護有沒有用」和「保護值不值得」分開了。通話系統的邪門歪道就在這裡：音質變好 0.1 MOS，但多了 40 ms mouth-to-ear delay，實際上可能是賠本生意。ITU 的 G.114 對 conversational speech 指出，one-way delay 應盡量低，低於約 150 ms 時多數應用的互動性近乎透明；400 ms 則是一般 network planning 的上限。([itu.int][2])

## 實驗管線要長這樣

不要只做：

`clean audio -> codec -> drop frame -> decode`

那太乾，像把魚拿去沙漠測游泳。

你要做的是：

`clean speech -> neural audio codec encode -> 20 ms packetization -> network impairment (delay/jitter/loss/burst loss) -> jitter buffer / late discard -> PLC / FEC / protection mechanism -> decode -> quality eval`

原因很直接：IETF 在 RTP/RTCP XR 裡把 **network loss** 和 **discard due to jitter** 明確分開；兩者都會傷害語音品質，但診斷意義不同。還有專門的 de-jitter buffer metrics、burst/gap discard metrics、post-repair loss metrics 可報。([datatracker.ietf.org][3])

## 你至少要比的 baseline

我建議 4 組，這樣論文味和工程味都夠：

1. **No protection baseline**
   完全不保護。

2. **Random protection baseline**
   在相同 redundancy budget 下，隨機選 frame 保護。

3. **Simple heuristic baseline**
   例如只用 energy / VAD / onset 做保護。

4. **Your importance-aware method**
   用你定義的 frame importance 分數決定保護對象。

更完整一點，再加：

5. **Oracle baseline**
   用完整 reference hindsight 算「這個 frame 掉了造成的真實品質下降」當上界。
   這不是可部署方法，但非常適合看你的 importance score 距離理想排序差多遠。

這裡的核心原則只有一條：**所有方法都要在同一個保護成本下比較**。
也就是相同額外 bit budget、相同冗餘比例、或相同每秒可保護 frame 數。
不然結果會變成「花更多錢所以比較好」，這種結論沒有靈魂。

## importance 應該怎麼切成可評估版本

先別一口氣把宇宙都塞進去。建議拆成三版：

**A. content-only importance**
只看編碼前後可取得的 frame 特徵，例如：

* voiced / unvoiced
* onset / transition
* short-term energy jump
* pitch stability
* 前後 frame embedding distance 或 token entropy

**B. content + history importance**
再加：

* 前 1–2 個 frame 是否剛遺失
* 是否在 silence→speech / word boundary
* PLC 是否剛做過 concealment

**C. content + history + network-state importance**
最後再加：

* 當前估計 jitter
* JBM buffer depth
* late-arrival risk
* burst-loss state

這樣你能回答一個很關鍵的問題：
**改善到底來自「音訊內容辨識」還是「網路狀態感知」？**

## 網路條件不要只做隨機 loss

這是很多 PoC 最容易翻車的地方。

你至少要做四種 scenario：

### 1) Random loss

例如 0.5%、1%、3%、5%、10%。

用途：先看最乾淨的 robustness 曲線。

### 2) Burst loss

用 Gilbert–Elliott 或等價兩態模型。
因為 fullband E-model 對 VoIP 不只考慮 packet loss，也考慮 **burst ratio**；burstiness 不是裝飾品，它真會改變感知品質。([ITU][4])

### 3) Jitter-only / late discard

封包不是丟了，是**太晚到**。
這種情況如果你只做 erase simulation，會完全漏掉。RTCP XR 明確把 lost 和 discarded due to jitter 分開定義。([IETF Datatracker][3])

### 4) Mixed delay + jitter + loss

最好直接吃 3GPP 的 MTSI/VoLTE delay-loss profiles。
TS 26.132 Annex E/F 就是專門給 MTSI speech / jitter buffer behavior 的 delay and loss profiles，甚至包含 real VoLTE capture profile 與 DRX 20 ms / 40 ms 類型條件。([ETSI][5])

---

### 我會建議的最小實驗矩陣

每個方法都跑：

* Random PLR：0.5 / 1 / 3 / 5%
* Burst loss：平均 PLR 1 / 3 / 5%，BurstR 至少兩級
* Jitter profile：輕 / 中 / 重
* Mixed MTSI profiles：至少 2 個

這樣你最後可以畫出：

* MOS vs PLR
* MOS vs BurstR
* MOS vs late discard rate
* MOS vs added delay

這四張圖會很有說服力。

## 音檔資料集不要只挑乾淨朗讀

IMS call 最怕的常常不是長穩態母音，而是**轉換點**。
像 silence→speech、子音爆破、詞邊界、重音切換，這些地方掉 frame 很容易讓 PLC 補得像靈魂出竅。

所以資料至少要分層：

* male / female
* quiet speech / expressive speech
* short utterance / long utterance
* lots of onset-transitions / relatively smooth speech
* 單講者為主；若有餘力再做 double-talk 或 background noise

然後每個檔案預先標註：

* speech onset
* voiced/unvoiced
* phoneme/word boundary（粗略也行）
* silence region

這樣你就能做 **“damage by region”** 分析：
importance-aware 是否真的特別保護了最痛的區域，而不是只是平均灑糖粉。

## 評估指標：分三層看

### 第一層：音質 / 可懂度

主指標我會建議：

* **MOS-LQO / POLQA 類指標**：作為主品質分數
* **STOI / ESTOI**：看 intelligibility
* **SI-SDR / LSD / ViSQOL 類**：當輔助，不要當唯一結論

P.863.1 有提醒，實際 network testing 往往是多種 degradation 同時存在，而且交互作用不一定線性，所以單一 objective metric 不應被神化成宇宙真理。([ITU][6])

### 第二層：通話系統指標

這層很 IMS：

* network loss rate
* late discard rate
* repaired loss count / post-repair loss count
* jitter buffer delay
* concealment rate
* consecutive concealment bursts
* end-to-end one-way delay / added playout delay

這些指標在 RTP XR / de-jitter / repair metrics 裡都有對應概念。([IETF Datatracker][7])

### 第三層：importance 方法本身的診斷指標

這層是你自己的靈魂指標：

* protected-frame precision@K
  被保護的 frame 中，有多少真的是高傷害 frame
* oracle ranking correlation
  與 oracle importance 的 Spearman correlation
* damage captured ratio
  前 K% 被保護 frame 覆蓋了多少總品質損失
* onset coverage / transition coverage
  重要轉換點有沒有被保護到

這些不屬於標準，但對你的方法論非常關鍵。

## 主觀測試怎麼做才像通話，不像音樂播放器試聽

P.800 明確區分了 **listening-only** 和 **conversational** 方法；開發早期可先用 listening-only，但如果你要說「符合 IMS audio call 用途」，最後一定要補一個小型 conversational 或近似 conversational 的主觀測試。([ITU][8])

我建議分兩段：

### Phase 1：大規模客觀評估

先把所有條件掃過，選出前 2–3 個方法。

### Phase 2：小型主觀評估

只測最關鍵條件，例如：

* clean + burst loss
* jitter-heavy late discard
* mixed VoLTE profile
* onset-heavy utterances

主觀測試不用一開始就很豪華。
先做 15–25 人的 A/B 或 CCR / DCR 類比較就很有用了。
重點是讓受試者聽到**對話型場景**，不是只聽單句朗讀。

## 成功條件要先寫死，不然後面會吵成一鍋粥

我會建議你的 PoC success criteria 長這樣：

1. **在相同 redundancy budget 下**，importance-aware 的 MOS-LQO / intelligibility 顯著優於 random baseline。
2. 優勢在 **burst loss** 與 **mixed jitter+loss** 條件下特別明顯。
3. 改善主要來自：

   * 減少高傷害 frame 遺失
   * 降低 onset / transition 區域失真
   * 降低長 concealment burst
4. **新增延遲可控**，最好讓總 mouth-to-ear 仍落在互動上合理的範圍，至少別為了救音質把通話變成回聲山谷。([ITU][2])

## 如果你想做得更像 EVS / IMS，而不是純學術玩具

你可以把保護機制切成兩型：

### 型 1：frame-level redundant copy

某些重要 frame 多送一次，或 piggyback 到後續 packet。

### 型 2：partial protection

不是整個 frame 重送，而是只保護關鍵 token / codebook / low-frequency latent。

這會更接近 EVS channel-aware / partial redundancy 的精神。EVS JBM 規範裡也討論了 partial redundancy / partial copies 與 FEC offset、late arrival、甚至因為晚到造成額外 playout delay 的處理。

這樣你就能回答下一個更有價值的問題：

**對 neural codec 而言，保護「整個 frame」比較划算，還是保護 frame 內最關鍵的一部分比較划算？**

這個問題很有料，已經不是玩具題了。

## 我最推薦的 PoC 版本

如果你今天就要開幹，我建議先做這版：

* frame = 20 ms
* 4 methods：none / random / heuristic / proposed
* 3 network types：random loss / burst loss / jitter+late discard
* 2 speech subsets：onset-heavy / general
* metrics：MOS-LQO、STOI、concealment rate、late discard rate、added delay
* 固定 redundancy budget：例如 10% 額外 payload
* 最後再做 20 組主觀 A/B

這版已經很像 IMS audio call evaluate，而不是只是「我 drop 一些 frame 然後聽起來好像還行」。

---

最關鍵的一句收尾：

**如果你的實驗沒有把 jitter buffer 和 late discard 納進來，那它評估的是 codec robustness，不是 IMS call robustness。**

這兩者長得像雙胞胎，但其實脾氣差很多。

你若需要，我下一步可以直接幫你把這套內容整理成一份 **可執行的實驗 spec**，包含：

* test matrix
* 每個 log 欄位
* 統計表格格式
* 以及最後論文/提案要畫的圖表清單。


