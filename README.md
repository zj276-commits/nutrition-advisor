# Nutrition Advisor — Multi-Agent AI System

A personal nutrition advisor powered by a multi-agent LLM pipeline. The user describes what they ate in plain language, and the system automatically looks up nutritional content, identifies nutrient deficiencies, and recommends specific foods to fill those gaps.

---

## System Overview

This nutrition advisor system acts as a personal nutritionist. The user describes what they ate in plain language (e.g., *"I had a hamburger and french fries for lunch, and a coffee with milk for breakfast"*), and the system automatically looks up the nutritional content of each food item, identifies which nutrients are lacking compared to daily recommended intake, and suggests specific foods to eat to fill those gaps. The goal is to help users make informed dietary decisions by combining real-time nutrition data with a curated food recommendation database.

The system chains three agents together in a pipeline. **Agent 1** handles function calling — it uses a custom tool function `search_usda_food()` that calls the USDA FoodData Central API to retrieve detailed nutrition data (calories, protein, fat, carbs, vitamins, minerals) for each food the user mentions. The API returns real, accurate nutrition values per 100g serving. **Agent 2** performs multi-agent analysis — it receives the nutrition table from Agent 1 and compares it against standard Recommended Daily Allowance (RDA) values embedded in its system prompt, then identifies the top 3 nutrient deficiencies. **Agent 3** uses RAG (Retrieval-Augmented Generation) with vector embeddings — it encodes all 35 foods in the local CSV database (`nutrition_foods.csv`) into vector representations using the `nomic-embed-text` embedding model, then performs semantic search via cosine similarity to find which foods are most relevant to the identified nutrient gaps. The LLM then generates practical, personalized food recommendations based on the retrieved context.

I chose the USDA FoodData Central API because it is free, reliable, and provides comprehensive nutrient breakdowns for thousands of foods. For the RAG component, I built a curated CSV of 35 common nutrient-dense foods across categories (vegetables, fruits, fish, meat, dairy, grains, legumes, nuts) rather than using a massive database, because the small local LLM (`smollm2:1.7b`) works better with concise, focused context. The retrieval uses `nomic-embed-text` to generate 768-dimensional vector embeddings for each food's full nutrient profile, then ranks candidates by cosine similarity against natural language queries — this is true semantic search, not just column sorting. One challenge was that the small model sometimes struggles with complex tool-calling for multiple foods in a single prompt, so I designed the workflow to parse the food list in R code and call the USDA API directly for each item, keeping the LLM's job focused on analysis and recommendation rather than API orchestration. Another challenge was handling the USDA API's nested JSON response — the tool function parses this into a clean `data.frame` so downstream agents receive simple tabular data they can reason about effectively.

---

## System Architecture

```
User Input (plain language meal description)
        │
        ▼
┌──────────────────────────────────────────┐
│  Agent 1: Food Nutrition Lookup          │
│  (Function Calling / Tool Use)           │
│                                          │
│  • Parses food items from user input     │
│  • Calls search_usda_food() for each     │
│  • Returns nutrition data table          │
│    (calories, protein, fat, carbs,       │
│     vitamins, minerals per 100g)         │
└──────────────┬───────────────────────────┘
               │ nutrition_data (data.frame)
               ▼
┌──────────────────────────────────────────┐
│  Agent 2: Nutrient Gap Analysis          │
│  (Multi-Agent Orchestration)             │
│                                          │
│  • Receives nutrition table from Agent 1 │
│  • Compares against RDA values           │
│  • Identifies top 3 deficiencies         │
└──────────────┬───────────────────────────┘
               │ gap analysis text
               ▼
┌──────────────────────────────────────────┐
│  Agent 3: Food Recommendations           │
│  (RAG — Vector Embedding Retrieval)      │
│                                          │
│  • Embeds nutrition_foods.csv using      │
│    nomic-embed-text model                │
│  • Semantic search via cosine similarity │
│  • Retrieves top 5 foods per nutrient    │
│  • LLM generates recommendations        │
│    using retrieved context               │
└──────────────┬───────────────────────────┘
               │
               ▼
    Personalized food recommendations
```

### Agent Roles

| Agent | Role | Input | Output | Technique |
|-------|------|-------|--------|-----------|
| Agent 1 | Food Nutrition Lookup | User's meal description | Nutrition data table (data.frame) | **Function Calling** — calls USDA FoodData Central API |
| Agent 2 | Nutrient Gap Analyst | Nutrition data table from Agent 1 | Top 3 nutrient deficiencies | **Multi-Agent Orchestration** — LLM reasoning with RDA context |
| Agent 3 | Food Recommender | Gap analysis + RAG-retrieved food data | 3-5 food recommendations | **RAG** — vector embedding retrieval (nomic-embed-text + cosine similarity) from local CSV, then generates |

---

## RAG Data Source

**File:** `data/nutrition_foods.csv`

The RAG component uses a curated CSV database of **35 nutrient-dense foods** spanning 8 categories:

| Category | Example Foods |
|----------|--------------|
| Vegetables | Spinach, Kale, Broccoli, Sweet Potato, Carrot, Bell Pepper, Tomato, Mushrooms |
| Fruits | Orange, Banana, Blueberries, Avocado, Strawberries, Kiwi |
| Fish | Salmon, Tuna, Sardines |
| Meat | Chicken Breast, Beef Lean, Liver Beef |
| Dairy | Greek Yogurt, Milk Whole, Cheese Cheddar |
| Grains | Quinoa, Brown Rice, Oats, Fortified Cereal |
| Legumes | Lentils, Chickpeas, Edamame |
| Nuts/Seeds | Almonds, Walnuts, Sunflower Seeds |
| Protein | Eggs, Tofu |

Each food has **14 nutrient columns**: calories, protein, fat, carbs, fiber, vitamin A, vitamin C, vitamin D, calcium, iron, potassium, vitamin B12, and zinc.

### Semantic Retrieval Process

The RAG search uses **vector embeddings** rather than simple column sorting:

1. **Build descriptions**: Each food row is converted into a rich text description (e.g., *"Spinach (vegetable, 1 cup raw) - calories: 7, protein: 0.9g, vitamin A: 469mcg, ..."*)
2. **Embed**: Both the food descriptions and the natural-language query are embedded using `nomic-embed-text` via Ollama's `embed()` function
3. **Cosine similarity**: The system computes the dot product of normalized vectors to rank foods by semantic relevance
4. **Retrieve**: The top N most relevant foods are returned with similarity scores as JSON context for the LLM

This approach allows natural language queries like *"foods rich in vitamin C for immune health"* instead of rigid column names like `"vitamin_c_mg"`.

---

## Tool Functions

| Tool Name | Purpose | Parameters | Returns |
|-----------|---------|------------|---------|
| `search_usda_food(query)` | Calls the USDA FoodData Central API to retrieve detailed nutrition data for a food item | `query` (string): food item name, e.g. "banana", "chicken breast" | A `data.frame` with 1 row containing: food name, calories, protein_g, fat_g, carbs_g, fiber_g, vitamin_a_mcg, vitamin_c_mg, vitamin_d_mcg, calcium_mg, iron_mg, potassium_mg, vitamin_b12_mcg |
| `search_nutrition(query, document, top_n, embed_model)` | RAG semantic retrieval — embeds a natural language query and all food descriptions using `nomic-embed-text`, then ranks by cosine similarity | `query` (string): natural language query, e.g. "foods rich in vitamin C"; `document` (string): path to CSV; `top_n` (integer): number of results; `embed_model` (string): embedding model name | JSON string containing the top N semantically matched foods with similarity scores, food_name, category, serving, and all nutrient values |
| `build_food_descriptions(db)` | Converts each food row into a rich text description for embedding | `db` (data.frame): the nutrition database | Character vector of food descriptions combining name, category, serving, and all nutrient values |
| `agent(messages, model, output, tools, all)` | Core agent wrapper — runs an LLM chat with optional tool calling | `messages`: list of chat messages; `model`: LLM model name; `output`: output format; `tools`: optional tool metadata; `all`: return all results or last only | LLM response text, or tool call results if tools are provided |
| `agent_run(role, task, tools, output, model)` | Convenience wrapper — creates a system+user message pair and runs the agent | `role` (string): system prompt; `task` (string): user message content; `tools`: optional tool list; `output`: format; `model`: LLM model | LLM response in the specified output format |
| `df_as_text(df)` | Converts a data.frame to markdown table text for LLM consumption | `df`: a data.frame | A markdown-formatted table string |

---

## Technical Details

### Dependencies (R Packages)

| Package | Version | Purpose |
|---------|---------|---------|
| `ollamar` | latest | Interface to Ollama LLM (local model interaction) |
| `dplyr` | ≥1.0 | Data wrangling and filtering |
| `readr` | ≥2.0 | Reading CSV files |
| `stringr` | ≥1.4 | String operations |
| `httr2` | ≥1.0 | HTTP requests to USDA API |
| `jsonlite` | ≥1.8 | JSON parsing and serialization |
| `knitr` | ≥1.40 | Converting data.frames to markdown tables |

### API

- **USDA FoodData Central API**: `https://api.nal.usda.gov/fdc/v1/foods/search`
- Free to use; supports a `DEMO_KEY` for testing (rate-limited) or a personal API key from [https://fdc.nal.usda.gov/api-key-signup.html](https://fdc.nal.usda.gov/api-key-signup.html)

### Models (via Ollama)

- **Chat LLM**: `smollm2:1.7b` — a small local model for analysis and generation. Chosen for lightweight local execution; larger models (e.g., `llama3`, `mistral`) can be swapped in via the `MODEL` variable
- **Embedding Model**: `nomic-embed-text` (137M params, 768 dimensions) — used for RAG vector embeddings and cosine similarity search. Provides semantic understanding of food descriptions and nutrient queries

### File Structure

```
nutrition-advisor/
├── nutrition_advisor.R       # Main system file — runs the 3-agent pipeline
├── functions.R               # Agent orchestration helpers (agent, agent_run, df_as_text)
├── data/
│   └── nutrition_foods.csv   # RAG database — 35 nutrient-dense foods
├── README.md                 # This documentation file
└── LICENSE                   # MIT License
```

---

## Usage Instructions

### 1. Prerequisites

- **R** (≥ 4.0) installed: [https://cran.r-project.org/](https://cran.r-project.org/)
- **Ollama** installed and running: [https://ollama.com/](https://ollama.com/)
- Internet connection (for USDA API calls)

### 2. Install Ollama and Pull the Model

```bash
# Install Ollama from https://ollama.com/download
# Then pull both models:
ollama pull smollm2:1.7b
ollama pull nomic-embed-text
```

### 3. Install R Packages

Open R or RStudio and run:

```r
install.packages(c("dplyr", "readr", "stringr", "httr2", "jsonlite", "knitr"))

# Install ollamar from CRAN (or GitHub if needed)
install.packages("ollamar")
```

### 4. Configure API Key

The script requires a USDA FoodData Central API key. Get a free key:

1. Visit [https://fdc.nal.usda.gov/api-key-signup.html](https://fdc.nal.usda.gov/api-key-signup.html)
2. Sign up and copy your key
3. In `nutrition_advisor.R`, set `USDA_API_KEY = "your_key_here"`

### 5. Run the System

```bash
# Make sure Ollama is running first
ollama serve

# Run the nutrition advisor
Rscript nutrition_advisor.R
```

Or open `nutrition_advisor.R` in RStudio and source it.

### 6. Customize the Input

Edit line 121 in `nutrition_advisor.R` to change what the user ate:

```r
input = "I had a hamburger and french fries for lunch, and a coffee with milk for breakfast."
```

Also update the `foods` vector on line 134 to match:

```r
foods = c("hamburger", "french fries", "coffee with milk")
```

---

## Repository Links

- **Main system file (3-agent pipeline):** [`nutrition_advisor.R`](https://github.com/zj276-commits/nutrition-advisor/blob/main/nutrition_advisor.R)
- **Multi-agent orchestration & helper functions:** [`functions.R`](https://github.com/zj276-commits/nutrition-advisor/blob/main/functions.R)
- **RAG database:** [`data/nutrition_foods.csv`](https://github.com/zj276-commits/nutrition-advisor/blob/main/data/nutrition_foods.csv)

---

## License

MIT License — see [LICENSE](LICENSE) for details.
