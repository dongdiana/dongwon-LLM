# Greek Yogurt Purchase Decision Simulation

A sophisticated LLM-based simulation system that predicts Greek yogurt purchase decisions across diverse consumer personas using demographic and psychographic characteristics.

## Project Overview

This project leverages OpenAI's GPT-4o-mini model through LangChain to simulate consumer behavior and predict purchase decisions for Greek yogurt products. Each persona is independently evaluated based on their unique characteristics including region, gender, age, education, occupation, and household size.

### Key Features

- **Persona-Based Simulation**: Simulate purchase decisions for various consumer personas
- **Asynchronous Processing**: Efficient batch processing with configurable concurrency
- **Comprehensive Logging**: Detailed logging and progress tracking
- **Flexible Data Loading**: Support for JSONL persona files and JSON product data
- **Robust Error Handling**: Graceful handling of API failures and invalid responses
- **Detailed Reporting**: Generate comprehensive simulation reports and statistics

## Project Structure

```
dongwon/
├── data/
│   ├── persona_info/           # Persona data (JSONL files)
│   │   └── *.jsonl            # Persona characteristics files
│   └── product_info/          # Product market information
│       └── 그릭요거트.json     # Greek yogurt market data
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
1. Load persona data from `data/persona_info/` (creates sample data if none exists)
2. Load Greek yogurt market information
3. Run LLM simulations for each persona
4. Generate and save detailed results

### Persona Data Format

Create JSONL files in `data/persona_info/` with the following format:

```jsonl
{"id": "0", "region": "서울", "gender": "남자", "age": "25", "education": "대학교 졸업", "occupation": "회사원", "household_size": "1인 가구"}
{"id": "1", "region": "경기도", "gender": "여자", "age": "35", "education": "대학원 졸업", "occupation": "전문직", "household_size": "3인 가구"}
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
```

#### Prompt Customization (`prompt.yaml`)

- **`system_prompt`**: Instructions for the LLM persona simulation
- **`user_prompt_template`**: Template with persona characteristics
- **`minimal_prompt_template`**: Fallback for incomplete persona data

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
- **`0`**: Will not purchase Greek yogurt
- **`1`**: Will purchase Greek yogurt

## Technical Implementation

### Data Loading (`src/loader.py`)

- **`PersonaLoader`**: Loads and validates JSONL persona files
- **`ProductLoader`**: Processes Greek yogurt market data
- Robust error handling and data validation

### LLM Simulation (`src/simulator.py`)

- **`PersonaSimulator`**: Core simulation engine
- Asynchronous processing with configurable concurrency
- Intelligent prompt formatting based on available persona attributes
- Comprehensive error handling and retry logic

### Key Features

1. **Adaptive Prompting**: Uses full or minimal templates based on available persona data
2. **Batch Processing**: Configurable concurrent request limits
3. **Progress Tracking**: Real-time progress bars using tqdm
4. **Robust Parsing**: Intelligent extraction of purchase decisions from LLM responses
5. **Comprehensive Logging**: Detailed logs for debugging and monitoring

## Error Handling

The system handles various error scenarios:

- **Missing API Key**: Clear error message with setup instructions
- **Invalid Persona Data**: Validation and warning messages
- **API Failures**: Retry logic with exponential backoff
- **Malformed Responses**: Robust parsing with fallback strategies
- **File I/O Errors**: Graceful handling with informative error messages

## Sample Data

If no persona files are found, the system automatically creates sample data with 5 diverse personas for demonstration purposes.

## Monitoring & Debugging

### Log Files

- **Console Output**: Real-time progress and status updates
- **`simulation.log`**: Detailed file-based logging for debugging

### Statistics Tracking

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

## License

This project is proprietary software developed for Greek yogurt market research purposes.

## Support

For technical support or questions about the simulation results, please refer to the log files and error messages for detailed diagnostics.
