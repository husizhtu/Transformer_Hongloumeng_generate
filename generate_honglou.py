import torch
import random
import textwrap  # 自动换行
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.functional import dropout

# 超参数
device = "cuda" if torch.cuda.is_available() else "cpu"
block_size = 256  # seq长度
batch_size = 64  # 批次处理数
n_embd = 384  # 词嵌入维度
num_heads = 8
head_size = n_embd // num_heads
n_layer = 6
learning_rate = 0.0003
max_iters = 1000
eval_interval = int(max_iters / 10)
eval_iters = 200
dropout_value = 0.2
torch.manual_seed(1337)  # 随机种子
file_name = "hong_lou_meng.txt"
wrap_width = 100

# -----数据预处理-----
with open(file_name, 'r', encoding='utf-8') as f:
    text = f.read()

# 字典并有序化
chars = sorted(list(set(text)))
vocab_size = len(chars)

# 投影
s2i = {ch: i for i, ch in enumerate(chars)}  # 符号投影到整数
i2s = {i: ch for i, ch in enumerate(chars)}  # 整数投影到字符
encode = lambda str1: [s2i[c] for c in str1]  # 字符串转化为数字串，即编码
decode = lambda list1: "".join([i2s[i] for i in list1])  # 数字串转化为字符串，即解码

# 训练、验证分组
data = torch.tensor(encode(text), dtype=torch.long)  # 整数表示字符
n = int(0.9 * len(data))  # 前90%用于训练
train_data = data[:n]  # 训练
val_data = data[n:]  # 验证
print(train_data)
print(val_data)
print(f'文件{file_name}读取完成')


# 批次化
def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))  # 防溢出
    x = torch.stack([data[i:i + block_size] for i in ix])  # 输入
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])  # 输出(target)
    x, y = x.to(device), y.to(device)
    return x, y


# -----损失评测-----
@torch.no_grad()
def estimate_loss(model):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


# Head类（其实就是串联前文）
class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)  # 线性变换层
        self.query = nn.Linear(n_embd, head_size, bias=False)  # 线性变换层
        self.value = nn.Linear(n_embd, head_size, bias=False)  # 线性变换层
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))  # 不可训练的结构，即常量下三角矩阵
        self.dropout = nn.Dropout(dropout_value)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)  # (B,T,head_size)
        q = self.query(x)  # (B,T,head_size)

        wei = q @ k.transpose(-2, -1) * k.shape[-1] ** 0.5  # 注意力方阵
        wei = wei.masked_fill(self.tril == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)

        v = self.value(x)
        out = wei @ v
        return out


# 多头注意力
class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout_value)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


# 残差化
class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, n_embd * 4),
            nn.ReLU(),
            nn.Linear(n_embd * 4, n_embd),
            nn.Dropout(dropout_value)
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, n_embd, num_heads):
        super().__init__()
        self.sa = MultiHeadAttention(num_heads, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))  # 残差多头注意力
        x = x + self.ffwd(self.ln2(x))  # 残差前馈
        return x


# -----语言模型-----
class LanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, num_heads) for _ in range(n_layer)])  # 多级残差多头注意力
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape  # B=batch_size,T=block_size(其实就是时序)
        token_embd = self.token_embedding_table(idx)
        position_idx = torch.arange(T, device=device)
        position_embd = self.position_embedding_table(position_idx)
        x = token_embd + position_embd  # (B,T,n_embd)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        # head_out=self.multihead(x)
        # logits = self.network2(torch.relu(self.network1(head_out)))  # (B,T,vocab_size)
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, token_sequ, max_new_tokens):  # token_sequ是已知的上文，max_new_tokens是续写的长度
        for _ in range(max_new_tokens):
            tokens_input = token_sequ[:, -block_size:]
            logits, loss = self.forward(tokens_input)  # logits:(B,T,vocab_size)
            logits = logits[:, -1, :]  # -1表示只取最后一个（概率分布格式）
            probs = F.softmax(logits, dim=-1)
            token_next = torch.multinomial(probs, num_samples=1)  # 概率分布向量-->one_hot向量-->整数
            token_next = token_next.to(token_sequ.device)
            token_sequ = torch.cat((token_sequ, token_next), dim=1)
        new_tokens = token_sequ[:, -max_new_tokens:]
        return new_tokens


# -----运行-----
if __name__ == '__main__':
    print(f"训练内容：{file_name}")
    model = LanguageModel()  # 实例化
    model = model.to(device)
    print(sum(p.numel() for p in model.parameters()) / 1e6, 'M parameters')  # 打印参数数量

    # 设定优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    # 训练循环
    for i in range(max_iters):
        if i % eval_interval == 0 or i == max_iters - 1:
            losses = estimate_loss(model)
            print(f"step {i}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

        # 取样
        xb, yb = get_batch("train")
        logits, loss = model(xb, yb)  # 前馈运算
        optimizer.zero_grad(set_to_none=True)  # 旧梯度归零
        loss.backward()  # 反向传播，计算新梯度
        optimizer.step()  # 进一步优化计算

    print("训练结束，开始生成内容")
    max_new_tokens = 500
    start_idx = random.randint(0, len(val_data) - block_size - max_new_tokens)

    # 上文内容
    context = torch.zeros((1, block_size), dtype=torch.long, device=device)  # (B,T) B=1,T=block_size
    context[0, :] = val_data[start_idx:start_idx + block_size]
    context_str = decode(context[0].tolist())
    wrapped_context_str = textwrap.fill(context_str, width=wrap_width)

    # 真实下文
    real_next_tokens = torch.zeros((1, max_new_tokens), dtype=torch.long, device=device)
    real_next_tokens[0, :] = val_data[start_idx + block_size:start_idx + block_size + max_new_tokens]
    real_next_tokens_str = decode(real_next_tokens[0].tolist())
    wrapped_real_next_tokens = textwrap.fill(real_next_tokens_str, width=wrap_width)

    # 生成下文
    generated_tokens = model.generate(context, max_new_tokens)
    generated_str = decode(generated_tokens[0].tolist())
    wrapped_generated_str = textwrap.fill(generated_str, width=wrap_width)

    print("上文内容：")
    print(wrapped_context_str)
    print("生成内容：")
    print(wrapped_generated_str)
    print("真实下文：")
    print(wrapped_real_next_tokens)

    # x, y = get_batch("train")
    # print(x)
    # x_list = x.tolist()
    # for str_list in x_list:
    #     decode_str = decode(str_list)
    #     print(decode_str)
    # token_embedding_table = nn.Embedding(vocab_size, n_embd, device=device)
    # embd = token_embedding_table(x)
    # position_embedding_table = nn.Embedding(block_size, n_embd, device=device)
    # position_idx = torch.arange(block_size).to(device)
    # position_embd = position_embedding_table(position_idx)
    # print(embd)
    # print(position_embd)
