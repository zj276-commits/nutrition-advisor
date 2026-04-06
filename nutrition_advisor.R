# homework2_nutrition_advisor.R
# Nutrition Advisor: Multi-Agent System with RAG and Function Calling
# Aaron
#
# This script builds an AI nutrition advisor that:
# 1. Takes a user's meal description and looks up nutrition info via the USDA API (Function Calling)
# 2. Analyzes nutrient gaps against recommended daily intake (Multi-Agent)
# 3. Recommends foods to fill the gaps using a local nutrition database (RAG)

# 0. SETUP #################################

## 0.1 Load Packages ############################

library(ollamar) # for LLM interaction
library(dplyr)   # for data wrangling
library(readr)   # for reading CSV
library(stringr) # for string operations
library(httr2)   # for HTTP API requests
library(jsonlite) # for JSON parsing

# Source the agent() and agent_run() helper functions
source("functions.R")

## 0.2 Configuration ############################

MODEL = "smollm2:1.7b"
EMBED_MODEL = "nomic-embed-text"
NUTRITION_DB = "data/nutrition_foods.csv"
USDA_API_KEY = "hDd8SEbJXKiWhU8ifgeAJJOGQipK6bjelZzDJzLL"

# 1. TOOL DEFINITION (Function Calling) #################################

# This tool calls the USDA FoodData Central API to look up nutrition info
# for a given food item. It parses the complex response into a clean summary.
search_usda_food = function(query) {
  # Build request to USDA FoodData Central API
  req = request("https://api.nal.usda.gov/fdc/v1/foods/search") %>%
    req_url_query(query = query, api_key = USDA_API_KEY, pageSize = 1) %>%
    req_headers(Accept = "application/json") %>%
    req_method("GET")

  # Perform request and parse response
  resp = req %>% req_perform()
  data = resp_body_json(resp)

  # If no results found, return a message
  if (length(data$foods) == 0) {
    return(data.frame(food = query, note = "Not found in USDA database"))
  }

  # Extract the first matching food item
  food = data$foods[[1]]
  nutrients = food$foodNutrients

  # Parse nutrients into a clean data.frame
  # We want: calories, protein, fat, carbs, fiber, vitamin A/C/D, calcium, iron
  target_ids = c(1008, 1003, 1004, 1005, 1079, 1106, 1162, 1114, 1087, 1089, 1092, 1098)
  target_names = c("calories", "protein_g", "fat_g", "carbs_g", "fiber_g",
                   "vitamin_a_mcg", "vitamin_c_mg", "vitamin_d_mcg",
                   "calcium_mg", "iron_mg", "potassium_mg", "vitamin_b12_mcg")

  result = data.frame(food = food$description, stringsAsFactors = FALSE)

  for (i in seq_along(target_ids)) {
    # Find the matching nutrient
    value = NA
    for (n in nutrients) {
      if (!is.null(n$nutrientId) && n$nutrientId == target_ids[i]) {
        value = n$value
        break
      }
    }
    result[[target_names[i]]] = ifelse(is.null(value), 0, value)
  }

  return(result)
}

# Define tool metadata so the LLM knows how to call it
tool_search_usda_food = list(
  type = "function",
  "function" = list(
    name = "search_usda_food",
    description = "Search the USDA FoodData Central database for nutrition information about a food item. Returns calories, protein, fat, carbs, fiber, vitamins, and minerals per 100g.",
    parameters = list(
      type = "object",
      required = list("query"),
      properties = list(
        query = list(type = "string", description = "The food item to search for, e.g. 'banana', 'chicken breast', 'french fries'")
      )
    )
  )
)


# 2. RAG SEARCH FUNCTION (Vector Embedding + Cosine Similarity) #################################

# Build a text description of each food row for embedding.
# Combines name, category, and all nutrient values into a single string
# so the embedding captures nutritional meaning, not just the food name.
build_food_descriptions = function(db) {
  paste0(
    db$food_name, " (", db$category, ", ", db$serving, ") - ",
    "calories: ", db$calories, ", ",
    "protein: ", db$protein_g, "g, ",
    "fat: ", db$fat_g, "g, ",
    "carbs: ", db$carbs_g, "g, ",
    "fiber: ", db$fiber_g, "g, ",
    "vitamin A: ", db$vitamin_a_mcg, "mcg, ",
    "vitamin C: ", db$vitamin_c_mg, "mg, ",
    "vitamin D: ", db$vitamin_d_mcg, "mcg, ",
    "calcium: ", db$calcium_mg, "mg, ",
    "iron: ", db$iron_mg, "mg, ",
    "potassium: ", db$potassium_mg, "mg, ",
    "vitamin B12: ", db$vitamin_b12_mcg, "mcg"
  )
}

# Semantic search over the nutrition database using vector embeddings.
# Instead of sorting by a single column, this embeds the query and all food
# descriptions, then ranks by cosine similarity for true semantic retrieval.
search_nutrition = function(query, document = NUTRITION_DB, top_n = 5, embed_model = EMBED_MODEL) {
  db = read_csv(document, show_col_types = FALSE)

  descriptions = build_food_descriptions(db)

  # Embed all food descriptions and the query using Ollama
  food_embeddings = embed(embed_model, descriptions)
  query_embedding = embed(embed_model, query)

  # Cosine similarity (embed() normalizes by default, so dot product = cosine sim)
  similarities = as.vector(crossprod(food_embeddings, query_embedding))

  top_idx = order(similarities, decreasing = TRUE)[1:top_n]

  results = db[top_idx, ]
  results$similarity_score = round(similarities[top_idx], 4)

  cat("  RAG query:", query, "\n")
  cat("  Top match:", results$food_name[1], "(similarity:", results$similarity_score[1], ")\n")

  results %>%
    select(food_name, category, serving, similarity_score, everything()) %>%
    as.list() %>%
    jsonlite::toJSON(auto_unbox = TRUE)
}


# 3. MULTI-AGENT WORKFLOW #################################

# Simulate user input: what they ate today
input = "I had a hamburger and french fries for lunch, and a coffee with milk for breakfast."

cat("========================================\n")
cat("  Nutrition Advisor\n")
cat("========================================\n")
cat("User input:", input, "\n\n")

## Agent 1: Food Nutrition Lookup (Function Calling) ################
# This agent uses the USDA API tool to look up nutrition data
cat("--- Agent 1: Looking up nutrition data via USDA API ---\n")

# Since the small LLM may struggle with complex tool calls for multiple foods,
# we parse the foods manually and call the tool directly for reliability
foods = c("hamburger", "french fries", "coffee with milk")

# Look up each food via the USDA API
nutrition_data = do.call(rbind, lapply(foods, function(f) {
  cat("  Searching USDA for:", f, "\n")
  search_usda_food(f)
}))

# Show what Agent 1 found
cat("\nAgent 1 Result - Nutrition data retrieved:\n")
print(nutrition_data)

# Convert to text for Agent 2
nutrition_text = df_as_text(nutrition_data)

## Agent 2: Nutrient Gap Analysis (Multi-Agent) ################
cat("\n--- Agent 2: Analyzing nutrient gaps ---\n")

# Provide RDA context in the role so the agent can compare
role2 = paste(
  "You are a nutrition analyst. The user provides a table of foods they ate today with nutrient values per 100g.",
  "Compare their intake against these daily recommended values:",
  "Calories: 2000, Protein: 50g, Fat: 65g, Carbs: 300g, Fiber: 25g,",
  "Vitamin A: 900mcg, Vitamin C: 90mg, Vitamin D: 15mcg,",
  "Calcium: 1000mg, Iron: 18mg, Potassium: 2600mg, Vitamin B12: 2.4mcg.",
  "List the TOP 3 nutrients the user is most likely deficient in.",
  "Be brief: just list the 3 nutrients and why they are lacking. Use bullet points."
)

result2 = agent_run(role = role2, task = nutrition_text, model = MODEL, output = "text")

cat("Agent 2 Result - Nutrient gap analysis:\n")
cat(result2, "\n")


## Agent 3: Food Recommendations via RAG ################
cat("\n--- Agent 3: Recommending foods via RAG (semantic search) ---\n")

# Use vector-embedding RAG to search for foods matching the identified gaps.
# These are natural language queries — the embedding model understands the
# semantic meaning and finds relevant foods, not just exact column matches.
rag_result1 = search_nutrition("foods rich in vitamin C for immune health")
rag_result2 = search_nutrition("foods high in calcium for strong bones")
rag_result3 = search_nutrition("high fiber foods for digestive health")

# Combine RAG-retrieved context
rag_context = paste(
  "Foods rich in Vitamin C:", rag_result1,
  "\nFoods high in Calcium:", rag_result2,
  "\nFoods high in Fiber:", rag_result3
)

role3 = paste(
  "You are a friendly nutrition advisor.",
  "The user ate a meal today that is low in certain nutrients.",
  "Based on the nutrient gap analysis and the food database provided,",
  "recommend 3-5 specific foods or snacks they could eat today to fill the gaps.",
  "Be practical and specific. Format as a short bullet-point list.",
  "The user's nutrient gap analysis and available food recommendations are provided below."
)

task3 = paste("Nutrient gaps:\n", result2, "\n\nAvailable nutrient-rich foods:\n", rag_context)
result3 = agent_run(role = role3, task = task3, model = MODEL, output = "text")

cat("Agent 3 Result - Food recommendations:\n")
cat(result3, "\n")

cat("\n========================================\n")
cat("  Workflow Complete!\n")
cat("========================================\n")
