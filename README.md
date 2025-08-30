# Product Purchase Decision Simulation

A sophisticated LLM-based simulation system that predicts product purchase decisions across diverse consumer personas using demographic and psychographic characteristics. The system is designed to work with various food products and consumer goods.

## Project Overview

This project leverages OpenAI's GPT-4o-mini model through LangChain to simulate consumer behavior and predict purchase decisions for various products. The system supports two simulation types:

- **Type A**: Single-question simulations using basic demographic personas
- **Type B**: Multi-question sessions with detailed persona characteristics and product choice scenarios

Each persona is independently evaluated based on their unique characteristics including gender, age, household size, income level, and detailed psychographic profiles.

## Project Structure

```
dongwon/
├── data/
│   ├── persona.csv           # Basic persona characteristics (Type A)
│   ├── product_info/         # Product market information
│   │   └── 그릭요거트.json     # Product market data with target & similar products
│   └── naver_trend/          # Naver search trend data
│       └── 그릭요거트.json     # Search trend data by gender and age
├── persona/                  # Detailed persona profiles (Type B)
│   └── 그릭요거트.json          # Product-specific detailed personas
├── src/
│   ├── loader.py             # Data loading utilities & product option generation
│   ├── simulator.py          # LLM simulation engine with multi-question support
│   └── report.py             # Result analysis and reporting
├── results/                  # Simulation output files
│   ├── simulation_results_*.json
│   └── summary_report_*.json
├── config.yaml              # Model and simulation configuration
├── prompt.yaml              # LLM prompts and templates (A & B types)
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
1. Load persona data based on simulation type (CSV for Type A, JSON for Type B)
2. Load product market information and generate randomized options (Type B)
3. Run LLM simulations for each persona with automatic rate limiting
4. Execute single or multi-question sessions based on configuration
5. Generate and save detailed results with demographic analysis

## Rate Limiting

To prevent API rate limit issues, the system automatically applies pacing:

- **5-second break** after every 10 requests
- **60-second break** after every 100 requests

This ensures stable processing and prevents OpenAI API throttling. The system will log these breaks in the console for transparency.

## Simulation Types

### Type A: Single Question Simulation

**Persona Data**: Create CSV files in `data/` with the following format:

```csv
id,성별,연령대,가구원수,소득구간
16115,여성,40대,3인가구 이상,중간소득(월 300~700만원 미만)
19821,여성,60대,2인가구,저소득(월 300만원 미만)
6526,남성,30대,3인가구 이상,중간소득(월 300~700만원 미만)
```

**Response Format**: `0` (No) or `1` (Yes) with reasoning for single purchase decision.

### Type B: Multi-Question Simulation

**Persona Data**: Create detailed JSON files in `persona/{product}.json`:

```json
[
  {
    "uuid": "917822",
    "segment_key_input": "고소득(월 700만원 이상)-만 50~59세-3인가구 이상-남성",
    "reasoning": "이 페르소나는 고소득 남성으로, 건강과 영양에 대한 관심이 높습니다...",
    "가구소득": "고소득(월 700만원 이상)",
    "연령대": "만 50~59세",
    "성별": "남성",
    "건강관심도": "매우 그렇다",
    "우유구입기준": "영양(건강)",
    ...
  }
]
```

**Question Flow**:
1. **Product Selection**: Choose from randomized list of target product + similar products
2. **Quantity Selection**: How many units to purchase (if product selected)

**Response Format**: Structured responses with product choices and quantities.

## Configuration Options

#### LLM Settings (`config.yaml`)

```yaml
llm:
  model_name: "gpt-4o-mini"    # OpenAI model
  temperature: 0.3             # Response creativity
  timeout: 30                  # API timeout seconds

simulation:
  batch_size: 10               # Concurrent requests
  save_detailed_logs: true     # Enable detailed logging

# Product Configuration
product:
  filename: "그릭요거트"         # Product info filename (auto-adds .json)

# Prompt Configuration  
prompts:
  type: "B"                    # Simulation type: "A" or "B"
```

#### Prompt Customization (`prompt.yaml`)

**Type A Prompts**:
- **`system_prompt_A`**: Basic demographic persona simulation
- **`user_prompt_A`**: Single purchase decision question

**Type B Prompts**:
- **`system_prompt_B`**: Detailed persona with reasoning and characteristics  
- **`user_prompt_B1`**: Product selection from randomized options
- **`user_prompt_B2`**: Quantity selection for chosen product

#### Product Configuration

To simulate different products, update the product configuration in `config.yaml`:

```yaml
product:
  filename: "product_name"  # Without .json extension
```

Then create a corresponding JSON file in `data/product_info/product_name.json` with:

**Type A Structure** (Market context only):
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

**Type B Structure** (Target + Similar products):
```json
{
  "덴마크 하이그릭요거트": {
    "product_info": {
      "content": ["Target product description..."]
    },
    "nutrition_per100": {
      "칼로리": "70 kcal",
      "단백질": "7 g",
      ...
    }
  },
  "유사제품군": {
    "풀무원 그릭요거트": {
      "product_info": ["Similar product description..."],
      "nutrition_per100": {...}
    },
    "그릭데이 시그니처": {
      "product_info": ["Another similar product..."],
      "nutrition_per100": {...}
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

**Type A Results**:
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
      "reasoning": "건강한 식품이라 구매하겠습니다",
      "raw_response": "1, 건강한 식품이라 구매하겠습니다",
      "response_time": 1.23,
      "success": true,
      "timestamp": "2024-01-15T10:30:01"
    }
  ]
}
```

**Type B Results**:
```json
{
  "metadata": {...},
  "results": [
    {
      "persona_id": "917822",
      "persona": {...},
      "product_options_order": ["덴마크 하이그릭요거트", "풀무원 그릭요거트", "그릭데이 시그니처"],
      "selected_product": "덴마크 하이그릭요거트",
      "selected_quantity": "2",
      "all_responses": [
        {
          "question": "user_prompt_B1",
          "response": "1, 높은 단백질 함량과 무첨가 특성이 마음에 듭니다",
          "selected_number": "1",
          "selected_product": "덴마크 하이그릭요거트",
          "reasoning": "높은 단백질 함량과 무첨가 특성이 마음에 듭니다"
        },
        {
          "question": "user_prompt_B2", 
          "response": "2, 가족과 함께 섭취하기 위해",
          "selected_quantity": "2",
          "reasoning": "가족과 함께 섭취하기 위해"
        }
      ],
      "response_time": 2.45,
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

## Response Formats

### Type A Response Format
Each persona simulation returns:
- **`0`**: Will not purchase the product  
- **`1`**: Will purchase the product
- **`reasoning`**: Explanation for the decision

Structured response format: `decision_number, reasoning_explanation`

### Type B Response Format
Multi-question session with:
- **Question 1**: Product selection from randomized options (`number, reasoning`)
- **Question 2**: Quantity selection (`quantity, reasoning`)
- **Results**: Selected product name, quantity, and complete response history

## Technical Implementation

### Data Loading (`src/loader.py`)

- **`PersonaLoader`**: 
  - Loads CSV personas (Type A) and detailed JSON personas (Type B)
  - Processes persona characteristics and reasoning
- **`ProductLoader`**: 
  - Processes product market data and nutrition information
  - Generates randomized product options for Type B simulations
  - Handles target products and similar product categories

### LLM Simulation (`src/simulator.py`)

- **`PersonaSimulator`**: Core simulation engine with dual-mode support
- **Type A**: Single-question purchase decision simulation
- **Type B**: Multi-question product selection and quantity simulation
- Asynchronous processing with configurable concurrency
- Intelligent prompt formatting based on simulation type
- Automatic **product market context** loading and **Naver search trend** extraction
- Advanced response parsing for both simple decisions and complex product choices
- Session management for multi-question workflows

### Reporting (`src/report.py`)

- **`SimulationReporter`**: Generates comprehensive analysis reports
- Handles both Type A and Type B result structures  
- Demographic breakdown analysis (gender, age groups)
- Automatic detection of simulation type for appropriate metrics
- Purchase rate calculations and statistical summaries

## Dependencies

- **LangChain**: LLM integration framework
- **OpenAI**: GPT model API access
- **PyYAML**: Configuration file parsing
- **python-dotenv**: Environment variable management
- **tqdm**: Progress bar visualization
- **aiohttp**: Asynchronous HTTP requests
- **pandas**: Data manipulation and analysis
