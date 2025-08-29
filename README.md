# Product Purchase Decision Simulation

A sophisticated LLM-based simulation system that predicts product purchase decisions across diverse consumer personas using demographic and psychographic characteristics. The system is designed to work with various food products and consumer goods.

## Project Overview

This project leverages OpenAI's GPT-4o-mini model through LangChain to simulate consumer behavior and predict purchase decisions for various products. Each persona is independently evaluated based on their unique characteristics including gender, age, household size, and income level.

## Project Structure

```
dongwon/
├── data/
│   ├── persona.csv           # Persona characteristics files
│   ├── product_info/         # Product market information
│   │   └── 그릭요거트.json     # Product market data (example: Greek yogurt)
│   └── naver_trend/          # Naver search trend data
│       └── 그릭요거트.json     # Search trend data by gender and age
├── src/
│   ├── loader.py             # Data loading utilities
│   └── simulator.py          # LLM simulation engine
├── results/                  # Simulation output files
│   ├── simulation_results_*.json
│   └── summary_report_*.json
├── config.yaml              # Model and simulation configuration
├── prompt.yaml              # LLM prompts and templates
├── main.py                  # Main execution script
├── requirements.txt         # Python dependencies
├── .env                     # Environment variables (OPENAI_API_KEY)
└── README.md               # Project documentation
```

## Installation & Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Environment Configuration

Create a `.env` file in the project root with your OpenAI API key:

```bash
OPENAI_API_KEY=your_openai_api_key_here
```

### 3. Configuration Files

The project includes pre-configured YAML files:

- **`config.yaml`**: LLM model settings, simulation parameters
- **`prompt.yaml`**: System and user prompt templates

## Usage

### Basic Execution

Run the complete simulation pipeline:

```bash
python main.py
```

The system will:
1. Load persona data from `data/persona.csv`
2. Load product market information based on configuration
3. Run LLM simulations for each persona with automatic rate limiting
4. Generate and save detailed results

## Rate Limiting

To prevent API rate limit issues, the system automatically applies pacing:

- **5-second break** after every 10 requests
- **60-second break** after every 100 requests

This ensures stable processing and prevents OpenAI API throttling. The system will log these breaks in the console for transparency.

### Persona Data Format

Create CSV files in `data/` with the following format:

```csv
id,성별,연령대,가구원수,소득구간
16115,여성,40대,3인가구 이상,중간소득(월 300~700만원 미만)
19821,여성,60대,2인가구,저소득(월 300만원 미만)
6526,남성,30대,3인가구 이상,중간소득(월 300~700만원 미만)
```

### Configuration Options

#### LLM Settings (`config.yaml`)

```yaml
llm:
  model_name: "gpt-4o-mini"    # OpenAI model
  temperature: 0.7             # Response creativity
  max_tokens: 500              # Response length limit
  timeout: 30                  # API timeout seconds

simulation:
  batch_size: 10               # Concurrent requests
  save_detailed_logs: true     # Enable detailed logging

# Product Configuration
product:
  filename: "그릭요거트"         # Product info filename (auto-adds .json)
```

#### Prompt Customization (`prompt.yaml`)

- **`system_prompt`**: Instructions for the LLM persona simulation
- **`user_prompt_template`**: Template for purchase questions

#### Product Configuration

To simulate different products, update the product configuration in `config.yaml`:

```yaml
product:
  filename: "product_name"  # Without .json extension
```

Then create a corresponding JSON file in `data/product_info/product_name.json` with:

```json
{
  "product_name": {
    "product_info": {
      "content": ["Product description and features..."]
    }
  },
  "market_report": {
    "content": ["Market analysis and context..."]
  }
}
```

**Examples:**
- Greek Yogurt: `filename: "그릭요거트"`
- Instant Noodles: `filename: "라면"`
- Energy Drinks: `filename: "에너지드링크"`

#### Naver Trend Data (Optional)

To include search trend analysis in the market context, add a corresponding JSON file in `data/naver_trend/{product}.json`:

```json
{
  "gender": {
    "2023-01-01": {"f": 60.48, "m": 44.65},
    "2023-02-01": {"f": 66.45, "m": 52.89},
    ...
  },
  "age": {
    "2023-01-01": {"20": 73.74, "30": 68.73, "40": 45.97, "50": 20.28, "60": 6.11},
    "2023-02-01": {"20": 67.23, "30": 69.73, "40": 51.60, "50": 24.94, "60": 8.12},
    ...
  }
}
```

The system will automatically:
- Load trend data if available
- Calculate cumulative shares by gender and age
- Add trend insights to the market context
- Support time-decay weighting (optional)

**Preprocessing Raw Trend Data:**

If you have raw Naver trend data, use the preprocessing script to convert it:

```bash
# Process a specific product
python data/naver_trend/preprocessing.py 그릭요거트

# Process all files in data/naver_trend/raw/
python data/naver_trend/preprocessing.py
```

This converts `data/naver_trend/raw/{product}.json` to `data/naver_trend/{product}.json` format.

## Output Files

### Simulation Results

**`results/simulation_results_YYYYMMDD_HHMMSS.json`**

```json
{
  "metadata": {
    "simulation_time": "2024-01-15T10:30:00",
    "model_used": "gpt-4o-mini",
    "total_personas": 100,
    "statistics": {...}
  },
  "results": [
    {
      "persona_id": "001",
      "persona": {...},
      "purchase_decision": "1",
      "raw_response": "1",
      "response_time": 1.23,
      "success": true,
      "timestamp": "2024-01-15T10:30:01"
    }
  ]
}
```

### Summary Report

**`results/summary_report_YYYYMMDD_HHMMSS.json`**

```json
{
  "total_personas": 100,
  "successful_simulations": 98,
  "failed_simulations": 2,
  "purchase_decisions": {
    "will_not_purchase": 45,
    "will_purchase": 53,
    "invalid_responses": 0
  },
  "purchase_rate": 0.54
}
```

## Response Format

Each persona simulation returns:
- **`0`**: Will not purchase the product
- **`1`**: Will purchase the product
- **`reasoning`**: Explanation for the decision

Structured response format: `decision_number, reasoning_explanation`

## Technical Implementation

### Data Loading (`src/loader.py`)

- **`PersonaLoader`**: Loads and validates CSV persona files
- **`ProductLoader`**: Processes product market data

### LLM Simulation (`src/simulator.py`)

- **`PersonaSimulator`**: Core simulation engine
- Asynchronous processing with configurable concurrency
- Intelligent prompt formatting based on available persona attributes
- Automatic **product market context** loading and **Naver search trend** extraction
- Includes search trend analysis by gender and age
- Structured response parsing for decision and reasoning

## Tracking

The system tracks comprehensive statistics:
- Total personas processed
- Success/failure rates
- Purchase decision distribution
- Processing times
- Error categories

## Dependencies

- **LangChain**: LLM integration framework
- **OpenAI**: GPT model API access
- **PyYAML**: Configuration file parsing
- **python-dotenv**: Environment variable management
- **tqdm**: Progress bar visualization
- **aiohttp**: Asynchronous HTTP requests
