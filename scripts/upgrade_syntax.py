import os
import re
from pathlib import Path

def upgrade_file(file_path):
    with open(file_path, 'r') as f:
        content = f.read()

    original_content = content

    # Replace typing imports
    # Simple replacements for generic types
    replacements = {
        r'List\[': 'list[',
        r'Dict\[': 'dict[',
        r'Set\[': 'set[',
        r'Tuple\[': 'tuple[',
        r'Type\[': 'type[',
    }

    for pattern, replacement in replacements.items():
        content = re.sub(pattern, replacement, content)

    # Replace A | B with A | B
    # This is a bit complex to do perfectly with regex, but we can handle simple cases
    # We'll look for A | B and try to replace it.
    # A better approach for Union and Optional is to use libcst or similar, but for this task
    # we will try a robust regex for common cases.
    
    # T | None -> T | None
    content = re.sub(r'Optional\[([^\]]+)\]', r'\1 | None', content)
    
    # A | B -> A | B
    # Handling nested brackets is hard with regex. We will do a best effort for non-nested ones first.
    # This regex matches A | B | C where A, B, C do not contain brackets
    content = re.sub(r'Union\[([^\[\]]+)\]', lambda m: ' | '.join([x.strip() for x in m.group(1).split(',')]), content)

    # Clean up imports
    # Remove List, Dict, Set, Tuple, Type, Optional, Union from typing imports
    # This is also best effort.
    
    lines = content.split('\n')
    new_lines = []
    for line in lines:
        if line.strip().startswith('from typing import'):
            # Remove deprecated types
            types_to_remove = ['List', 'Dict', 'Set', 'Tuple', 'Type', 'Optional', 'Union']
            for t in types_to_remove:
                line = re.sub(rf'\b{t}\b,?', '', line)
                line = re.sub(rf', \b{t}\b', '', line) # cleanup trailing commas
            
            # Clean up double commas and empty imports
            line = re.sub(r',\s*,', ', ', line)
            line = re.sub(r'import\s*,', 'import ', line)
            line = line.strip()
            if line.endswith(','):
                line = line[:-1]
            
            if line.strip() == 'from typing import':
                continue # Empty import line
                
        new_lines.append(line)
    
    content = '\n'.join(new_lines)

    if content != original_content:
        print(f"Upgrading {file_path}")
        with open(file_path, 'w') as f:
            f.write(content)

def main():
    root_dir = Path('.')
    for file_path in root_dir.rglob('*.py'):
        if 'venv' in str(file_path) or '.git' in str(file_path):
            continue
        upgrade_file(file_path)

if __name__ == '__main__':
    main()
