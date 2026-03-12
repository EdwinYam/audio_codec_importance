Here's my analysis and proposed alternatives to zero-padding. The key insight is that token 0 is an arbitrary codebook entry — it's not silence, so it injects random artifacts. Better strategies use information from neighboring frames.

Proposed Concealment Strategies (best to simplest)
1. Neighbor Copy (Last-Good-Frame Repetition)
The simplest effective PLC: repeat the tokens from the previous received frame. This works because adjacent speech frames are highly correlated. For leading lost frames (no previous), use the next available frame.

2. Linear Interpolation of Tokens
For isolated or short bursts of loss, interpolate between the last received frame before and the first received frame after the gap. Since RVQ tokens are discrete indices, this operates in the dequantized embedding space (for HILCodec) or simply copies the nearest neighbor (for token-domain codecs).

3. Weighted Neighbor Blend (Fade)
For burst losses, blend the last-good and next-good frames with a linear crossfade weight. The frame closer to a received boundary gets more weight from that boundary.

4. Median Token from Neighbors
Use a small window of surrounding received frames and pick the most common (mode) token per codebook position. Good for noisy/bursty environments.

Let me implement strategies 1 and 2 as a concealment parameter in decode_with_mask, since they give the biggest PESQ bang for the complexity: