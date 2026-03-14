import os
import sys

def generate_tree(dir_path, prefix='', exclude_dirs=None):
    if exclude_dirs is None:
        exclude_dirs = {'.git', 'venv', '__pycache__', '.idea', 'logs', '.pytest_cache'}
    
    try:
        entries = sorted(os.listdir(dir_path))
    except Exception:
        return ""
        
    entries = [e for e in entries if e not in exclude_dirs]
    
    result = []
    for i, entry in enumerate(entries):
        path = os.path.join(dir_path, entry)
        is_last = (i == len(entries) - 1)
        connector = '└── ' if is_last else '├── '
        result.append(f'{prefix}{connector}{entry}')
        
        if os.path.isdir(path):
            new_prefix = prefix + ('    ' if is_last else '│   ')
            result.extend(generate_tree(path, new_prefix, exclude_dirs))
    return result

if __name__ == '__main__':
    base_dir = r'd:\mt5_trading_bot'
    tree_lines = generate_tree(base_dir)
    with open('tree_output.txt', 'w', encoding='utf-8') as f:
        f.write(os.path.basename(os.path.abspath(base_dir)) + '\n')
        f.write('\n'.join(tree_lines))
