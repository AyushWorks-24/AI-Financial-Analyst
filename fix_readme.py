import re

# Read existing README
with open('README.md', 'r', encoding='utf-8', errors='replace') as f:
    content = f.read()

# Remove any existing metadata block
if content.startswith('---'):
    end = content.find('---', 3)
    content = content[end+3:].lstrip()

# Write new metadata + content with correct UTF-8 encoding
metadata = "---\ntitle: AI Financial Analyst\nemoji: \U0001F4C8\ncolorFrom: blue\ncolorTo: purple\nsdk: streamlit\nsdk_version: \"1.40.0\"\npython_version: \"3.10\"\napp_file: app.py\npinned: false\n---\n\n"

with open('README.md', 'w', encoding='utf-8') as f:
    f.write(metadata + content)

print("Done! First 200 chars:")
print(open('README.md', encoding='utf-8').read()[:200])