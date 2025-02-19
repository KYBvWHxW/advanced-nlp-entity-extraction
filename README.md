# ReAct: Synergizing Reasoning and Acting in Language Models

This is an implementation of the ReAct paper (ICLR 2023) that demonstrates the synergy between reasoning and acting in language models.

## Project Structure

```
react_paper_implementation/
├── config/              # Configuration files
├── data/               # Data files and datasets
├── src/               # Source code
├── tests/             # Test files
├── requirements.txt   # Python dependencies
└── README.md         # This file
```

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure MCP:
Create a file at `~/.codeium/windsurf/mcp_config.json` with the following content:
```json
{
  "mcpServers": {
    "wikipedia-api": {
      "command": "npx",
      "args": [
        "-y",
        "@modelcontextprotocol/server-wikipedia"
      ]
    }
  }
}
```

## Features

- Implements the ReAct paradigm combining reasoning and acting in language models
- Supports integration with external knowledge sources (Wikipedia API)
- Provides implementations for various tasks:
  - Question answering (HotpotQA)
  - Fact verification (Fever)
  - Text-based game (ALFWorld)
  - Webpage navigation (WebShop)

## Usage

[Coming soon]

## Citation

```bibtex
@inproceedings{yao2023react,
  title={ReAct: Synergizing Reasoning and Acting in Language Models},
  author={Yao, Shunyu and Zhao, Jeffrey and Yu, Dian and Du, Nan and Shafran, Izhak and Narasimhan, Karthik and Cao, Yuan},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2023}
}
```
