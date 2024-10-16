﻿# Medical-Chatbot

## Getting Started

### Prerequisites

- Python 3.11
- Poetry (Follow this (https://python-poetry.org/docs/#installation) to install Poetry on your system)

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/hackathon-iiitl/Medical-Chatbot.git
   cd Medical-Chatbot
   ```

2. Install dependencies using Poetry:

   ```bash
   poetry install --no-root
   ```

3. Set up your environment variables:

   - Rename the `.env.example` file to `.env` and update the variables inside with your own values. Example:

   ```bash
   mv .env.example .env
   ```

4. Activate the Poetry shell to run the examples:

   ```bash
   poetry shell
   ```

5. Run the code examples:

   ```bash
    python Chatbot/chatbot.py
   ```
