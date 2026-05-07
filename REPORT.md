# Lab 21 — Evaluation Report

**Học viên**: Lê Duy Anh — 2A202600094
**Ngày nộp**: <2026-05-07>
**Submission option**: B (HF Hub)

## 1. Setup

- **Base model**: `unsloth/Llama-3.2-3B-Instruct-bnb-4bit`
- **Dataset**: `5CD-AI/Vietnamese-alpaca-gpt4-gg-translated`, 200 samples (180 train + 20 eval)
- **max_seq_length**: 1024 (`p95 = 562`, rounded up and capped at 1024)
- **GPU**: Tesla T4, 15.8 GB VRAM
- **Training cost**: Estimated cost: $0.07
- **HF Hub link**: https://huggingface.co/Cole1111/lab21-qwen25-3b-lora-r16

## 2. Rank Experiment Results

| Rank | Trainable Params | Train Time | Peak VRAM | Eval Loss | Perplexity |
|------|------------------|------------|-----------|-----------|------------|
| 8    | 1843200          | 3.863050 min    | 7.216138 GB    | 1.557694       | 4.747861        |
| 16   | 3686400          | 3.988256 min    | 6.617750 GB    | 1.516083       | 4.554353        |
| 64   | 14745600         | 3.803988  min   | 7.998512 GB    | 1.476811       | 4.378959        |
| Base | -                | -               | -              | ...            | ...             |


## 3. Loss Curve Analysis

![Loss curve](./results/loss_curve.png)

Loss curve của rank `r=16` cho thấy xu hướng giảm tổng thể. Ban đầu training loss khoảng `1.61`, sau đó giảm dần xuống khoảng `1.39` ở các step cuối. Dù đường loss có dao động nhẹ ở một số đoạn, đặc biệt quanh step 35–40 và 50–55, xu hướng chung vẫn là giảm. Điều này cho thấy LoRA adapter đã học được pattern từ training dataset.

Final eval loss khoảng `1.516`, cao hơn final training loss. Đây là hiện tượng bình thường vì model thường fit tốt hơn trên train set so với eval set chưa thấy trong quá trình train. Khoảng cách giữa train loss và eval loss không quá lớn, nên chưa có dấu hiệu overfitting nghiêm trọng. Tuy nhiên, vì dataset tương đối nhỏ, model vẫn có thể overfit nếu train nhiều epoch hơn hoặc dùng rank quá lớn.

Nhìn chung, loss curve cho thấy quá trình training ổn định và có hiệu quả. Adapter học tốt dần qua các step, và eval loss cuối đủ hợp lý để dùng so sánh với các rank khác.
## 4. Qualitative Comparison (5 examples)

### Example 1

**Prompt**:  
Giải thích LoRA là gì bằng ngôn ngữ đơn giản cho sinh viên mới học AI. Trả lời ngắn gọn, có ví dụ dễ hiểu.

**Base**:  
LoRA là viết tắt của "Low-Rank Adaptation" và là một kỹ thuật có thể cải thiện hiệu suất của mô hình NLP bằng cách làm cho mô hình nhỏ hơn mà không mất đi nhiều thông tin quan trọng. LoRA hoạt động bằng cách gắn kết một số trọng số của mô hình lớn với một số trọng số nhỏ hơn.

**Fine-tuned (r=16)**:  
LoRA là một phương pháp cải thiện mô hình NLP bằng cách gắn thêm một số tham số học được vào trọng số của mô hình. Ví dụ, với một mô hình như GPT-2, LoRA thêm một phần nhỏ vào các trọng số thay vì huấn luyện lại toàn bộ mô hình.

**Nhận xét**: improved nhẹ. Fine-tuned model giải thích gần với ngôn ngữ đơn giản hơn và có ví dụ dễ hiểu hơn. Tuy nhiên, base model lại nêu đúng tên đầy đủ “Low-Rank Adaptation”, nên cải thiện không quá lớn.

---

### Example 2

**Prompt**:  
So sánh LoRA và QLoRA theo 4 tiêu chí: cách hoạt động, VRAM, tốc độ train, và trường hợp nên sử dụng. Trình bày dạng bảng Markdown.

**Base**:  
Base model tạo bảng Markdown để so sánh LoRA và QLoRA, nhưng nội dung còn dài và một số ý chưa chính xác. Phần giải thích về LoRA còn nhầm sang các khái niệm như bias hoặc module nhỏ trong mô hình NLU.

**Fine-tuned (r=16)**:  
Fine-tuned model cũng tạo bảng Markdown để so sánh LoRA và QLoRA. Output có cố gắng trình bày theo format bảng, nhưng bảng chưa đầy đủ 4 tiêu chí như prompt yêu cầu và một số nội dung kỹ thuật vẫn còn chưa chính xác.

**Nhận xét**: same / degraded nhẹ. Cả hai model đều cố gắng làm theo format bảng Markdown, nhưng chưa trả lời đủ 4 tiêu chí. Fine-tuned model không cải thiện rõ rệt ở prompt này.

---

### Example 3

**Prompt**:  
Một nhóm muốn xây dựng chatbot hỏi đáp tài liệu nội bộ công ty. Họ nên dùng prompt engineering, RAG hay fine-tuning? Giải thích lựa chọn theo từng trường hợp.

**Base**:  
Base model đề xuất dùng fine-tuning để xây dựng chatbot hỏi đáp tài liệu nội bộ công ty. Model giải thích rằng fine-tuning giúp tận dụng dữ liệu nội bộ và làm chatbot phù hợp hơn với tác vụ cụ thể.

**Fine-tuned (r=16)**:  
Fine-tuned model cũng đề xuất dùng fine-tuning. Model giải thích rằng fine-tuning giúp tối ưu mô hình cho tác vụ hỏi đáp tài liệu nội bộ và cải thiện khả năng trả lời.

**Nhận xét**: same / degraded. Cả hai output chưa thật sự tốt vì với bài toán hỏi đáp tài liệu nội bộ, RAG thường là lựa chọn phù hợp hơn khi cần truy xuất knowledge từ tài liệu. Đây là một case cho thấy fine-tuned model không phải lúc nào cũng cải thiện khả năng ra quyết định kỹ thuật.

---

### Example 4

**Prompt**:  
Hướng dẫn từng bước chuẩn bị dataset Alpaca format để fine-tune một mô hình ngôn ngữ. Bao gồm ví dụ JSON có instruction, input, output.

**Base**:  
Base model đưa ra các bước chuẩn bị dữ liệu Alpaca format và có nhắc đến các trường `instruction`, `input`, `output`. Output có ví dụ JSON nhưng phần trình bày còn hơi dài và chưa thật sự gọn.

**Fine-tuned (r=16)**:  
Fine-tuned model trả lời theo dạng các bước rõ ràng hơn. Model đưa ra danh sách các cặp dữ liệu gồm `instruction`, `input`, `output` và có ví dụ JSON minh họa đúng format Alpaca.

**Nhận xét**: improved. Fine-tuned model bám sát yêu cầu hơn, trình bày ngắn gọn hơn và phù hợp trực tiếp với nội dung lab. Đây là một trong những example cho thấy fine-tuning giúp model trả lời đúng format mong muốn hơn.

---

### Example 5

**Prompt**:  
Giải thích trade-off khi tăng LoRA rank từ r=8 lên r=16 và r=64. Nêu ảnh hưởng đến trainable parameters, VRAM, training time và chất lượng output.

**Base**:  
Base model giải thích rằng khi tăng LoRA rank thì số lượng trainable parameters thay đổi và ảnh hưởng đến quá trình train. Tuy nhiên, câu trả lời còn khá chung và chưa phân tích rõ từng yếu tố như VRAM, training time và chất lượng output.

**Fine-tuned (r=16)**:  
Fine-tuned model đề cập trực tiếp đến các yếu tố cần phân tích: trainable parameters, VRAM, thời gian đào tạo và chất lượng đầu ra. Model giải thích rằng tăng rank từ `r=8` lên `r=16` và `r=64` sẽ làm tăng số tham số cần train và có thể cải thiện chất lượng, nhưng đánh đổi bằng chi phí tài nguyên cao hơn.

**Nhận xét**: improved. Fine-tuned model bám sát prompt hơn vì nhắc đúng các tiêu chí được yêu cầu. Output phù hợp với mục tiêu của lab rank experiment hơn base model.

## 5. Conclusion về Rank Trade-off

Trong thí nghiệm này, tôi so sánh ba LoRA rank là `r=8`, `r=16` và `r=64` trên cùng base model, cùng dataset và cùng hyperparameters. Kết quả quantitative cho thấy khi tăng rank thì perplexity có cải thiện: `r=8` có perplexity `4.747861`, `r=16` giảm xuống `4.554353`, và `r=64` đạt tốt nhất với `4.378959`. Điều này cho thấy rank cao hơn giúp adapter có nhiều capacity hơn để học pattern trong dataset.

Tuy nhiên, nếu xét ROI thì `r=16` là lựa chọn hợp lý nhất. Lý do là `r=16` cải thiện rõ so với `r=8`, nhưng chỉ dùng `3,686,400` trainable parameters, trong khi `r=64` dùng tới `14,745,600` trainable parameters, tức gấp 4 lần `r=16`. Mức cải thiện perplexity từ `r=16` lên `r=64` có tồn tại, nhưng không lớn tương xứng với số lượng tham số tăng thêm. Đây là dấu hiệu của diminishing returns: tăng rank vẫn cải thiện quality, nhưng chi phí tăng nhanh hơn lợi ích nhận được.

Nếu deploy production trong môi trường giới hạn tài nguyên như Colab T4 hoặc GPU nhỏ, tôi sẽ chọn `r=16` vì cân bằng tốt giữa chất lượng, VRAM, adapter size và chi phí train. Nếu mục tiêu là đạt chất lượng tốt nhất bất kể chi phí, `r=64` là lựa chọn tốt hơn. Nhưng với bài lab này và dataset tương đối nhỏ, `r=16` có ROI tốt nhất.

## 6. What I Learned

- Tôi hiểu rõ hơn rằng LoRA không fine-tune toàn bộ model mà chỉ train một lượng nhỏ adapter parameters, giúp giảm đáng kể chi phí train và VRAM.
- Tăng LoRA rank có thể cải thiện perplexity, nhưng không phải lúc nào cũng đáng giá vì trainable parameters và adapter size tăng rất nhanh.
- Perplexity là metric hữu ích để so sánh quantitative, nhưng vẫn cần qualitative prompts để kiểm tra model có thật sự trả lời đúng format và đúng yêu cầu hay không.
- Với dataset nhỏ và GPU giới hạn, rank trung bình như `r=16` thường là lựa chọn thực tế hơn so với chọn rank lớn nhất.