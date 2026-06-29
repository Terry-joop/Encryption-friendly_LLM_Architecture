import torch

with open('convert.py', 'r') as f:
    code = f.read()

old_str = "weight[f'{prefix}attn.self_attention.query_key_value.weight']"
new_str = "torch.cat([weight[f'{prefix}attn.self_attention.query.weight'], weight[f'{prefix}attn.self_attention.key.weight'], weight[f'{prefix}attn.self_attention.value.weight']], dim=0)"

if old_str in code:
    code = code.replace(old_str, new_str)
    with open('convert.py', 'w') as f:
        f.write(code)
    print("✨ convert.py 패치 완료!")
else:
    print("이미 패치되었거나 해당 코드를 찾을 수 없습니다.")
