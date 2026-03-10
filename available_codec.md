以你這組條件來看：**causal / streaming、實際可碰到 ≤20 ms、已有 pretrained、計算量不要太肥、模型越小越好**，真正值得先看的一線名單其實不多。我會把 **HILCodec、AudioDec、EnCodec 24k causal、Lyra v2** 放在前排；**Mimi、SNAC、DAC** 則比較像「有亮點，但不完全符合你這個即時通話預算」的候選。([arXiv][1])

**1) HILCodec — 最像你要的“小而能打”選手**
HILCodec 是 streaming、卷積式、24 kHz、支援 1.5–9 kbps，encoder 每 **320 samples** 產生一個 feature；在 24 kHz 下這相當於約 **13.3 ms** 的步進。作者公開 repo 直接提供 **ONNX / PyTorch pretrained checkpoints**。更關鍵的是，它在論文的 side-by-side complexity comparison 裡，參數量只有 **9.58M**，比 EnCodec 的 **14.85M** 和 AudioDec 的 **23.27M** 都小，且在單執行緒 CPU 的 streaming 模擬下 **RTF > 1**，也就是可即時。這顆很適合你要做 modem / IMS / token robustness 這種需要小步進、低複雜度、可實驗改造的場景。([arXiv][1])

**2) AudioDec — 工程上很強的 streaming baseline**
AudioDec 的定位非常明確：**open-source、streamable、real-time neural audio codec**。官方 repo 直接寫它能做 **48 kHz mono speech、12.8 kbps**，而且解碼延遲約 **GPU 6 ms / CPU 10 ms**，還有 real-time streaming demo 和 pretrained models。缺點是它比較偏 **speech codec**，而且模型沒有 HILCodec 那麼瘦；HILCodec 論文中的比較給它 **23.27M params**。但如果你要的是「先跑出真的低延遲 streaming 系統」，AudioDec 很值得當第一個強 baseline。([GitHub][2])

**3) EnCodec 24k causal — 最成熟、最好接各種研究管線**
EnCodec 的 24 kHz mono 版本是 **causal**，官方與 Hugging Face 都明確說它有 **streaming encoder-decoder architecture**，而且 24 kHz causal model 支援 **1.5 / 3 / 6 / 12 / 24 kbps**。Hugging Face model card 也寫了 streamable 模式是「不切 1 秒 chunk、而是只做左側 padding」，很適合做持續串流。它的好處是 ecosystem 成熟、Transformers / AudioCraft 周邊多，拿來做 token importance、丟包模擬、LM 接軌都很方便。缺點是官方文件沒有像 AudioDec / Lyra 那樣直接把「端到端 20 ms 內」寫死；不過 HILCodec 的比較中，它的 streaming 模擬 **RTF 4.25**、參數量 **14.85M**，工程上仍是很穩的選擇。([Hugging Face][3])

**4) Lyra v2 — 如果你目標就是語音通話，這顆很對味**
Lyra v2 雖然不是典型「大家拿來做 audio LM tokenization」那一路，但對 **speech communication** 很有味道。Google 的 repo 直接寫：它**每 20 ms** 萃取一次特徵、支援 **3.2–9.2 kbps**，並且能在低階手機上 real time 跑，**processing latency 20 ms**。API 還明講 encoder 送進去的就是 **20 ms** 音訊片段；repo 內也有 `model_coeffs/` 與 `model_path` 用法，表示公開程式碼就是照著已附模型係數在跑。缺點是它比較偏 **speech-only**、整合上較偏 **C++/Bazel** 世界，不像 EnCodec 那麼 PyTorch 研究友善；但如果你的目標是 **IMS voice / telephony style**，它其實非常合理。([GitHub][4])

**一句話結論：**

* 你要 **最小、最像 edge/real-time 研究原型**：**HILCodec**
* 你要 **最穩的 streaming speech baseline**：**AudioDec**
* 你要 **最成熟、最好和各種 token/LLM pipeline 接起來**：**EnCodec 24k causal**
* 你要 **最 telephony / speech-call 導向**：**Lyra v2** 

如果你是站在你目前那種 **IMS audio / packet importance / piggyback FEC / modem robustness** 的脈絡，我的排序會是：

選擇以下一種codec就好，按照順序如果不行在挑下一個
**HILCodec > EnCodec 24k causal > Lyra v2**。
原因很樸素：HILCodec 小、streaming、步進短；AudioDec 低延遲證據最直白；EnCodec 生態最成熟；Lyra 很強但更像產品型 speech codec，不像前面三顆那麼方便魔改。([arXiv][1])


