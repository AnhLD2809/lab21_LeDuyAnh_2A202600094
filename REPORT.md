# Lab 21 — Evaluation Report

**Học viên**: Lê Duy Anh — 2A202600094
**Ngày nộp**: <2026-05-07>
**Submission option**: B (HF Hub)

## 1. Setup
- **Base model**: unsloth/Llama-3.2-3B-Instruct-bnb-4bit
- **Dataset**: <tên dataset>, <số samples> (X train + Y eval)
- **max_seq_length**: <số> (p95 = <số>, rounded up)
- **GPU**: <Tesla T4 / L4 / A100>, <X> GB VRAM
- **Training cost**: $<số> (~<phút> @ $<rate>/hr)
- **HF Hub link** (nếu Option B): https://huggingface.co/<username>/<adapter-name>

## 2. Rank Experiment Results

| Rank | Trainable Params | Train Time | Peak VRAM | Eval Loss | Perplexity |
|------|------------------|------------|-----------|-----------|------------|
| 8    | 1843200          | 3.863050 min    | 7.216138 GB    | 1.557694       | 4.747861        |
| 16   | 3686400          | 3.988256 min    | 6.617750 GB    | 1.516083       | 4.554353        |
| 64   | 14745600         | 3.803988  min   | 7.998512 GB    | 1.476811       | 4.378959        |
| Base | -                | -               | -              | ...            | ...             |


## 3. Loss Curve Analysis
[Đính kèm loss_curve.png]
- Quan sát: <có / không có overfitting? Lý do?>

## 4. Qualitative Comparison (5 examples)

### Example 1
**Prompt**: ...
**Base**: ...
**Fine-tuned (r=16)**: ...
**Nhận xét**: <improved / same / degraded?>

[... lặp lại 4 examples nữa ...]

## 5. Conclusion về Rank Trade-off

<Tối thiểu 100 từ. Trả lời 3 câu hỏi:>
- Rank nào cho ROI tốt nhất trên dataset này? Tại sao?
- Khi nào tăng rank không còn cải thiện perplexity (diminishing returns)?
- Recommendation: nếu deploy production, bạn chọn rank nào? Tại sao?

## 6. What I Learned
- <Bullet 1: insight cá nhân>
- <Bullet 2: insight cá nhân>
- <Bullet 3: optional>