# BERT-Question Answering

- train: DRCD + DRCD-backtrans
	- 15329 paragraphs, 26918 questions
- dev: DRCD + DRCD-backtrans
	- 1255 paragraphs, 2863 questions
- test: DRCD + ODSQA
	- 1606 paragraphs, 3504 questions

# Baselines

- Simple(0.45139)
- Medium(0.65792)
- Strong(0.78136)
- Boss(0.83091)

# Results

## Simple(0.53291)

运行所给代码

## Medium(0.70431)

将`doc_stride`改为32

```python
self.doc_stride = 32
```

增加`LR_scheduler`的效果并没有变好(0.66969)

```python
from transformers import get_linear_schedule_with_warmup
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=100, 	num_training_steps=total_steps)
···
optimizer.zero_grad()
scheduler.step()
```

## Strong(0.82292)

修改`preprocessing`部分的代码，助教代码是以答案为中心截取段落，修改为随机抽取含答案的片段

```python
start_min = max(0, answer_end_token - self.max_paragraph_len + 1)
start_max = min(answer_start_token, len(tokenized_paragraph) - self.max_paragraph_len)
start_max = max(start_min, start_max)
paragraph_start = random.randint(start_min, start_max + 1)
paragraph_end = paragraph_start + self.max_paragraph_len
```

更换模型

```python
model_name = "luhua/chinese_pretrain_mrc_macbert_large"
```

修改后处理代码，输出结果部分存在[UNK]，将其替换为原始文段

```python
...
 # start index 应该比 end index 小 / start index 应该比 paragraph_end 大 / end index 应该比 paragraph_start 小
if start_index > end_index or start_index < paragraph_start or end_index > paragraph_end:
	continue
...
# Result 中存在[UNK] 说明存在编码问题 需要修正
if '[UNK]' in answer:
    print('发现 [UNK]，这表明有文字无法编码, 使用原始文本')
#   print("Paragraph:", paragraph)
#   print("Paragraph:", paragraph_tokenized.tokens)
    print('--直接解码预测:', answer)
    #找到原始文本中对应的位置
    #检查origin_start的值
    if origin_start is None:
        print('origin_start is None')
    else:
        # 检查token_to_chars的返回值
        token_to_chars_result = paragraph_tokenized.token_to_chars(origin_start)
        if token_to_chars_result is None:
            print('token_to_chars returned None')
            print(f'origin st:{origin_start}')
            print(f'len para token:{len(paragraph_tokenized)}')
        else:
            raw_start =  paragraph_tokenized.token_to_chars(origin_start)[0]
            # 检查origin_end的值
            if origin_start is None:
                print('origin_end is None')
            else:
                # 检查token_to_chars的返回值
                token_to_chars_result = paragraph_tokenized.token_to_chars(origin_end)
                if token_to_chars_result is None:
                    print('token_to_chars returned None')
                    print(f'origin ed:{origin_end}')
                    print(f'len para token:{len(paragraph_tokenized)}')
                else:
                    raw_end =  paragraph_tokenized.token_to_chars(origin_end)[1]
                    raw_end = paragraph_tokenized.token_to_chars(origin_end)[1]
                    answer = paragraph[raw_start:raw_end]
                    print('--原始文本预测:',answer)
```

将训练轮数增加为5轮后，分数略有上升(0.82463)。
