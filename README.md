# Product Purchase Decision Simulation

A sophisticated LLM-based simulation system that predicts product purchase decisions across diverse consumer personas using demographic and psychographic characteristics. The system is designed to work with various food products and consumer goods.

## Project Overview

This project leverages OpenAI's GPT-4o-mini model through LangChain to simulate consumer behavior and predict purchase decisions for various products. The system supports four simulation types:

- **Type A**: Single-question simulations using basic demographic personas
- **Type B**: Multi-question sessions with detailed persona characteristics and product choice scenarios
- **Type C**: Product conversion analysis comparing target products with existing consumer preferences
- **Type D**: TypeD product selection using specialized product data structure

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
├── persona/                  # Detailed persona profiles (Type B & C)
│   └── 그릭요거트.json          # Product-specific detailed personas with existing product usage
├── src/
│   ├── loader.py             # Data loading utilities & product option generation
│   ├── simulator.py          # LLM simulation engine with multi-question support
│   └── report.py             # Result analysis and reporting
├── results/                  # Simulation output files
│   ├── simulation_results_*.json
│   └── summary_report_*.json
├── config.yaml              # Model and simulation configuration
├── prompt.yaml              # LLM prompts and templates (A, B & C types)
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
1. Load persona data based on simulation type (CSV for Type A, JSON for Type B & C)
2. Load product market information and generate options (Type B) or comparison data (Type C)
3. Run LLM simulations for each persona with automatic rate limiting
4. Execute single questions (Type A), multi-question sessions (Type B), or conversion analysis (Type C)
5. Generate and save detailed results with demographic analysis and conversion rates


## Persona Generation CLI (`generate_personas`)

> Move into the CLI folder first:
```bash
cd generate_persona
```

### Multiple Choice(MC) Mode

The **MC mode** generates personas based primarily on **overall market context** (demographics + market report) for a given product group.  
It is suitable for products where **broad consumer traits** are more relevant than individual prior brand usage.  

**Applicable product groups:** `참치액`, `참치캔`

```powershell
python .\generate_personas.py `
  --mode mc `
  --keywords 참치액 `
  --n_samples 1000 `
  --batch_size 10 `
  --model gpt-4o-mini `
  --temperature 0.2 `
  --log_level DEBUG `
  --log_prompts `
  --prompt_preview_chars 600
```

```powershell
python .\generate_personas.py `
  --mode mc `
  --keywords 참치캔 `
  --n_samples 1000 `
  --batch_size 10 `
  --model gpt-4o-mini `
  --temperature 0.2 `
  --log_level DEBUG `
  --log_prompts `
  --prompt_preview_chars 600
```

### SWAP(SW) Mode

The **SWAP mode** generates personas by combining **overall market context** with **prior product/brand usage patterns**.
This mode enriches personas with substitution/switching behavior and is more suitable for categories with **brand competition**.

**Applicable product groups:** `그릭요거트`, `편의점커피라떼`, `스팸`

```powershell
python .\generate_personas.py `
  --mode swap `
  --keywords 그릭요거트 `
  --n_samples 1000 `
  --batch_size 10 `
  --model gpt-4o-mini `
  --temperature 0.2 `
  --log_level DEBUG `
  --log_prompts `
  --prompt_preview_chars 600
```

```powershell
python .\generate_personas.py `
  --mode swap `
  --keywords 편의점커피라떼 `
  --n_samples 1000 `
  --batch_size 10 `
  --model gpt-4o-mini `
  --temperature 0.2 `
  --log_level DEBUG `
  --log_prompts `
  --prompt_preview_chars 600
```

```powershell
python .\generate_personas.py `
  --mode swap `
  --keywords 스팸 `
  --n_samples 1000 `
  --batch_size 10 `
  --model gpt-4o-mini `
  --temperature 0.2 `
  --log_level DEBUG `
  --log_prompts `
  --prompt_preview_chars 600
```



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

### Type C: Product Conversion Analysis

**Persona Data**: Uses detailed JSON files in `persona/{product}.json` (same as Type B):

```json
[
  {
    "uuid": "367957",
    "segment_key_input": "고소득(월 700만원 이상)-만 70세 이상-1인가구-남성",
    "reasoning": "이 페르소나는 고소득의 70대 남성으로, 건강에 대한 관심이 높고...",
    "기존사용제품": "스팸 25% 라이트",
    "가구소득": "고소득(월 700만원 이상)",
    "연령대": "만 70세 이상",
    "성별": "남성",
    ...
  }
]
```

**Analysis Focus**: 
- **Target Product**: New product being promoted (first product in product_info JSON)
- **Current Product**: Persona's existing product preference from `기존사용제품` field
- **Conversion Question**: Whether persona would switch from current to target product

**Response Format**: `0` (No conversion) or `1` (Yes, will convert) with reasoning for product switching decision.

### Type D: TypeD Product Selection

**Persona Data**: Uses detailed JSON files in `persona/{product}.json` (same as Type B & C):

```json
[
  {
    "uuid": "367957",
    "segment_key_input": "고소득(월 700만원 이상)-만 70세 이상-1인가구-남성",
    "reasoning": "이 페르소나는 고소득의 70대 남성으로, 건강에 대한 관심이 높고...",
    "가구소득": "고소득(월 700만원 이상)",
    "연령대": "만 70세 이상",
    "성별": "남성",
    ...
  }
]
```

**Product Data**: Uses specialized TypeD structure in `data/product_info/TypeD/{product}.json`:

```json
{
  "동원맛참 고소참기름": {
    "category": ["동원맛참 고소참기름 90g", "동원맛참 고소참기름 135g"],
    "product_info": {
      "content": [
        "'동원맛참 고소참기름'은 참기름과 특제 소스로 간을 맞춰 별도 조리 없이 밥과 바로 먹는 '2세대 참치캔'이다.",
        "'동원맛참 고소참기름'은 90g·135g 두 규격으로 판매되며...",
        ...
      ],
      "출처": ["https://example.com/..."]
    }
  }
}
```

**Key Features**:
- **Product Name**: Extracted from top-level schema key ("동원맛참 고소참기름")
- **Product Info**: Uses `product_info.content` array (excluding `출처`)
- **Product Options**: Uses `category` array as selection choices
- **Data Source**: `data/product_info/TypeD/` folder instead of regular `data/product_info/`

**Response Format**: Selection from category options with reasoning for choice.

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

# Persona Configuration
persona:
  # Type A: Uses CSV file from data/ directory
  filename: "persona"          # CSV filename in data/ (.csv extension auto-added) - Only used for Type A
  
  # Type B & C: Uses JSON file from persona/ directory
  # File path: persona/{product.filename}.json - Uses product filename automatically
  
  sample_size: 0               # Number of personas to randomly sample (0 = use all personas)

# Product Configuration
product:
  filename: "그릭요거트"         # Product info filename (auto-adds .json)

# Prompt Configuration  
prompts:
  type: "D"                    # Simulation type: "A", "B", "C", or "D"
```

#### Persona File Configuration

**Important:** Persona loading works differently for each simulation type:

- **Type A**: Uses CSV file from `data/{persona.filename}.csv`
  - Example: `data/persona.csv` 
  - Contains basic demographic data (id, 성별, 연령대, 가구원수, 소득구간)

- **Type B, C & D**: Uses JSON file from `persona/{product.filename}.json`
  - Example: `persona/그릭요거트.json`
  - Contains detailed persona profiles with reasoning and characteristics
  - **Note:** The `persona.filename` setting is ignored for Types B, C & D

#### Random Sampling

Set `persona.sample_size` to control how many personas are used:
- `0`: Use all available personas (default)
- `>0`: Randomly sample N personas from the available set
- Works for all simulation types (A, B, C, D)

#### Prompt Customization (`prompt.yaml`)

**Type A Prompts**:
- **`system_prompt_A`**: Basic demographic persona simulation
- **`user_prompt_A`**: Single purchase decision question

**Type B Prompts**:
- **`system_prompt_B`**: Detailed persona with reasoning and characteristics  
- **`user_prompt_B1`**: Product selection from randomized options
- **`user_prompt_B2`**: Quantity selection for chosen product

**Type C Prompts**:
- **`system_prompt_C`**: Detailed persona with reasoning and characteristics (same as Type B)
- **`user_prompt_C`**: Product conversion analysis comparing target vs current product

**Type D Prompts**:
- **`system_prompt_D`**: Detailed persona with reasoning and characteristics
- **`user_prompt_D`**: TypeD product selection from category options

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

## Type C Conversion Analysis Usage

### Setting Up Type C Simulation

1. **Configure simulation type** in `config.yaml`:
```yaml
prompts:
  type: "C"
```

2. **Ensure persona data includes existing product usage**:
```json
{
  "uuid": "367957",
  "기존사용제품": "스팸 25% 라이트",
  // ... other persona fields
}
```

3. **Product data structure** in `data/product_info/{product}.json`:
```json
{
  "리챔 오믈레햄": {  // Target product (first key)
    "product_info": {"content": "..."},
    "nutrition_per100": {"칼로리": "241 kcal", ...}
  },
  "유사제품군": {
    "스팸 25% 라이트": {  // Current products (in similar products)
      "product_info": ["..."],
      "nutrition_per100": {"칼로리": "315 kcal", ...}
    },
    "리챔 더블라이트": {
      "product_info": ["..."],
      "nutrition_per100": {"칼로리": "230 kcal", ...}
    }
  }
}
```

### Example Conversion Analysis Output

```
=== CONVERSION ANALYSIS BY CURRENT PRODUCT ===
스팸 25% 라이트:
  Total Valid: 45
  Conversions: 20 (44.4%)
  No Conversions: 25
  Conversion Rate: 44.44%

리챔 더블라이트:
  Total Valid: 53  
  Conversions: 36 (67.9%)
  No Conversions: 17
  Conversion Rate: 67.92%
```

This analysis helps identify:
- Which existing products have users most likely to convert
- Overall market penetration potential for the target product
- Demographic patterns in conversion behavior

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

**Type C Results**:
```json
{
  "metadata": {...},
  "results": [
    {
      "persona_id": "367957",
      "persona": {...},
      "current_product": "스팸 25% 라이트",
      "target_product_info": "리챔 오믈레햄: product_info: ..., nutrition_per100: ...",
      "current_product_info": "스팸 25% 라이트: product_info: ..., nutrition_per100: ...",
      "conversion_decision": "1",
      "reasoning": "리챔 오믈레햄의 낮은 나트륨 함량과 편의성이 매력적입니다",
      "raw_response": "1, 리챔 오믈레햄의 낮은 나트륨 함량과 편의성이 매력적입니다",
      "response_time": 1.45,
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

**Type C Summary Report**:
```json
{
  "simulation_type": "C",
  "total_personas": 100,
  "successful_simulations": 98,
  "failed_simulations": 2,
  "conversion_decisions": {
    "no_conversion": 42,
    "conversion": 56,
    "invalid_responses": 0
  },
  "overall_conversion_rate": 0.57,
  "conversion_by_current_product": {
    "스팸 25% 라이트": {
      "no_conversion": 25,
      "conversion": 20,
      "total_valid": 45,
      "conversion_rate": 0.44,
      "conversion_percentage": 44.4
    },
    "리챔 더블라이트": {
      "no_conversion": 17,
      "conversion": 36,
      "total_valid": 53,
      "conversion_rate": 0.68,
      "conversion_percentage": 67.9
    }
  },
  "demographic_breakdown": {
    "by_gender": {...},
    "by_age": {...}
  }
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

### Type C Response Format
Product conversion analysis with:
- **Conversion Decision**: `0` (No conversion) or `1` (Yes, will convert)
- **Current Product**: Persona's existing product preference
- **Target Product**: New product being analyzed for market penetration
- **Results**: Conversion decision with detailed reasoning and product comparison data

## Technical Implementation

### Data Loading (`src/loader.py`)

- **`PersonaLoader`**: 
  - Loads CSV personas (Type A) and detailed JSON personas (Type B)
  - Processes persona characteristics and reasoning
- **`ProductLoader`**: 
  - Processes product market data and nutrition information
  - Generates randomized product options for Type B simulations
  - Extracts target and current product information for Type C conversion analysis
  - Handles target products and similar product categories

### LLM Simulation (`src/simulator.py`)

- **`PersonaSimulator`**: Core simulation engine with triple-mode support
- **Type A**: Single-question purchase decision simulation
- **Type B**: Multi-question product selection and quantity simulation
- **Type C**: Product conversion analysis comparing target vs current products
- Asynchronous processing with configurable concurrency
- Intelligent prompt formatting based on simulation type
- Automatic **product market context** loading and **Naver search trend** extraction
- Advanced response parsing for both simple decisions and complex product choices
- Session management for multi-question workflows

### Reporting (`src/report.py`)

- **`SimulationReporter`**: Generates comprehensive analysis reports
- Handles Type A, B, and C result structures  
- Demographic breakdown analysis (gender, age groups)
- Conversion rate analysis by current product (Type C)
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
